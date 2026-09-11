"""Render compact SOICT tables from downloaded Modal result.json files."""
from __future__ import annotations
import glob, json
from pathlib import Path

rows=[]
for path in glob.glob('artifacts/experiments/*/seed_*/result.json'):
    d=json.loads(Path(path).read_text())
    if d.get('status')!='complete': continue
    c=d['config']; t=d['test']
    rows.append((c['name'],c['seed'],t['cer'],t['wer'],t['exact_accuracy']))
rows.sort()
print('\\begin{tabular}{lrrr}')
print('\\toprule')
print('Configuration & CER (\\%) & WER (\\%) & Exact (\\%) \\\\')
print('\\midrule')
for n,s,cer,wer,ex in rows:
    label=n.replace('_', r'\_')
    print(f'{label} (s{s}) & {100*cer:.2f} & {100*wer:.2f} & {100*ex:.2f} \\\\')
print('\\bottomrule\n\\end{tabular}')
