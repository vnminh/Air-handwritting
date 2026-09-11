from __future__ import annotations

from collections import Counter, defaultdict

import torch


def greedy_decode(log_probs: torch.Tensor, tokens: list[str], blank: int = 0) -> list[str]:
    # log_probs: time, batch, classes
    sequences = log_probs.argmax(dim=-1).transpose(0, 1).tolist()
    decoded = []
    for sequence in sequences:
        previous = None
        output = []
        for index in sequence:
            if index != previous and index != blank:
                output.append(tokens[index])
            previous = index
        decoded.append("".join(output))
    return decoded


def edit_alignment(reference: list[str], hypothesis: list[str]):
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    operation = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], operation[i][0] = i, "D"
    for j in range(1, cols):
        cost[0][j], operation[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(cost[i - 1][j - 1], "C")]
            else:
                choices = [(cost[i - 1][j - 1] + 1, "S")]
            choices.extend([(cost[i - 1][j] + 1, "D"), (cost[i][j - 1] + 1, "I")])
            cost[i][j], operation[i][j] = min(choices, key=lambda item: item[0])
    aligned = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = operation[i][j]
        if op in {"C", "S"}:
            aligned.append((op, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            aligned.append((op, reference[i - 1], ""))
            i -= 1
        else:
            aligned.append((op, "", hypothesis[j - 1]))
            j -= 1
    return list(reversed(aligned))


def corpus_metrics(references: list[str], hypotheses: list[str]) -> dict:
    char_ops = Counter()
    word_ops = Counter()
    confusions = Counter()
    by_char = defaultdict(Counter)
    by_length = defaultdict(lambda: Counter(samples=0, errors=0, chars=0))
    exact = 0
    for reference, hypothesis in zip(references, hypotheses):
        exact += int(reference == hypothesis)
        alignment = edit_alignment(list(reference), list(hypothesis))
        for op, source, target in alignment:
            char_ops[op] += 1
            if source:
                by_char[source][op] += 1
            if op != "C":
                confusions[(source or "<ins>", target or "<del>")] += 1
        for op, _, _ in edit_alignment(reference.split(), hypothesis.split()):
            word_ops[op] += 1
        length = len(reference.replace(" ", ""))
        bucket = "2-3" if length <= 3 else "4-5" if length <= 5 else "6-7" if length <= 7 else "8+"
        by_length[bucket]["samples"] += 1
        by_length[bucket]["chars"] += len(reference)
        by_length[bucket]["errors"] += sum(op != "C" for op, _, _ in alignment)

    char_total = sum(char_ops.values()) - char_ops["I"]
    word_total = sum(word_ops.values()) - word_ops["I"]
    return {
        "cer": (char_ops["S"] + char_ops["D"] + char_ops["I"]) / max(char_total, 1),
        "wer": (word_ops["S"] + word_ops["D"] + word_ops["I"]) / max(word_total, 1),
        "exact_accuracy": exact / max(len(references), 1),
        "substitutions": char_ops["S"],
        "deletions": char_ops["D"],
        "insertions": char_ops["I"],
        "reference_characters": char_total,
        "top_confusions": [
            {"reference": source, "prediction": target, "count": count}
            for (source, target), count in confusions.most_common(20)
        ],
        "per_character": {
            char: {
                "count": sum(ops.values()) - ops["I"],
                "errors": ops["S"] + ops["D"],
                "error_rate": (ops["S"] + ops["D"]) / max(sum(ops.values()) - ops["I"], 1),
            }
            for char, ops in sorted(by_char.items())
        },
        "by_length": {
            bucket: {**counts, "cer": counts["errors"] / max(counts["chars"], 1)}
            for bucket, counts in by_length.items()
        },
    }

