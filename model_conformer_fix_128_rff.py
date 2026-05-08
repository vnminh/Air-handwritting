import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import make_splprep, interp1d
from scipy.signal import savgol_filter

BLANK = "-"

RFF_GAMMA = 5.0
RFF_NUM_FREQ = 10
INPUT_DIM = RFF_NUM_FREQ*2+6
_rff_rng = np.random.default_rng(42)
RFF_W = (_rff_rng.normal(0.0, np.sqrt(2.0 * RFF_GAMMA), size=(2, RFF_NUM_FREQ))).astype(np.float32)

def load_language(path: str):
    tokens = [BLANK]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(line)

    token2idx = {tok: i for i, tok in enumerate(tokens)}
    num_classes = len(tokens)
    return tokens, token2idx, num_classes


def encode_label(processed_label: str, token2idx: dict):
    parts = processed_label.split(BLANK)
    if len(parts)>1:
        parts = parts[1:-1]
    indices = [token2idx[part] for part in parts]
    return indices

# ---------------------------------------------------------------------------
# 2. Feature Extraction
# ---------------------------------------------------------------------------
def resample_by_arc_length_fixed(points: np.ndarray, num_points=256) -> np.ndarray:
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-9:
        return points

    target_lengths = np.linspace(0, total_length, num_points)

    interp_x = interp1d(cumulative_length, points[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(cumulative_length, points[:, 1], kind='linear', fill_value='extrapolate')

    resampled = np.column_stack([interp_x(target_lengths), interp_y(target_lengths)])
    return resampled

def normalize_coordinates(coord: np.ndarray, filter_size: int = 5, num_points:int=128) -> np.ndarray:
    """Normalize coordinates with arc length resampling using fixed density.
    
    Args:
        coord: (N, 2) array of raw coordinates
        filter_size: Window size for Savitzky-Golay filter
        arc_density: Target distance between points after resampling (in normalized units)
    """
    points = coord.astype(np.float64)

    # Step 1: Savitzky-Golay filter to reduce noise (applied per axis)
    if len(points) >= filter_size:
        polyorder = min(3, filter_size - 1)
        window = filter_size if filter_size % 2 == 1 else filter_size + 1
        points[:, 0] = savgol_filter(points[:, 0], window_length=window, polyorder=polyorder)
        points[:, 1] = savgol_filter(points[:, 1], window_length=window, polyorder=polyorder)

    # Step 2: Remove duplicate consecutive points
    dup = np.concatenate([[True], np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-6])
    unique_pts = points[dup]

    # Step 3: Smooth with spline (need >= 4 unique points for cubic spline)
    if len(unique_pts) >= 4:
        spline, t = make_splprep(unique_pts.T, s=len(unique_pts) * 2)
        smoothed = spline(t).T
    else:
        smoothed = unique_pts

    # Step 4: Shift to origin
    xmin, ymin = np.min(smoothed, axis=0)
    shifted = smoothed - np.array([xmin, ymin])

    # Step 5: Scale to [0, 1] preserving aspect ratio
    xmax, ymax = np.max(shifted, axis=0)
    # scale_factor = max(xmax, ymax)
    scale_factor = max(xmax, ymax)
    if scale_factor > 0:
        scaled = shifted / scale_factor
    else:
        scaled = shifted

    resampled = resample_by_arc_length_fixed(scaled, num_points=num_points)


    # return scaled.astype(np.float32)
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Data Augmentation (applied on raw coordinates before normalization)
# ---------------------------------------------------------------------------

def augment_rotation(coord: np.ndarray, angle_range=(-15, 15)) -> np.ndarray:
    """Rotate coordinates around centroid by a random angle (degrees)."""
    angle = np.random.uniform(*angle_range)
    rad = np.radians(angle)
    center = coord.mean(axis=0)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return ((coord - center) @ rot.T + center).astype(np.float32)


def augment_gaussian(coord: np.ndarray, std=2.0) -> np.ndarray:
    """Add Gaussian noise to each coordinate point."""
    noise = np.random.normal(0, std, coord.shape)
    return (coord + noise).astype(np.float32)


def augment_stretch(coord: np.ndarray, x_range=(0.8, 1.2), y_range=(0.8, 1.2)) -> np.ndarray:
    """Non-uniform scaling along x and y axes around centroid."""
    sx = np.random.uniform(*x_range)
    sy = np.random.uniform(*y_range)
    center = coord.mean(axis=0)
    return ((coord - center) * np.array([sx, sy]) + center).astype(np.float32)


def augment_time_warp(points, sigma=0.2):
    T = len(points)
    
    warp = np.linspace(0, 1, T)
    
    noise = np.random.normal(0, sigma, T)
    warp += noise
    
    warp = np.clip(warp, 0, 1)
    warp = np.sort(warp)
    
    idx = (warp * (T - 1)).astype(int)
    
    return points[idx]


def encode_label(processed_label: str, token2idx: dict):
    parts = processed_label.split(BLANK)
    if len(parts)>1:
        parts = parts[1:-1]
    indices = [token2idx[part] for part in parts]
    return indices

# ---------------------------------------------------------------------------
# 2. Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(sample: np.ndarray):
    eps = 1e-9

    # Random Fourier features (RBF approximation) from x, y coordinates.
    rff_phase = sample @ RFF_W
    rff = np.concatenate([np.cos(rff_phase), np.sin(rff_phase)], axis=1)

    # First derivative: delta x, delta y and normalized writing direction.
    diff1 = np.diff(sample, axis=0, prepend=sample[:1, :])
    diff1_norm = np.linalg.norm(diff1, axis=1, keepdims=True)
    direction = diff1 / (diff1_norm + eps)

    # Second derivative: normalized curvature direction.
    diff2 = np.diff(diff1, axis=0, prepend=diff1[:1, :])
    diff2_norm = np.linalg.norm(diff2, axis=1, keepdims=True)
    curvature = diff2 / (diff2_norm + eps)

    features = np.concatenate([
        rff,         # (T, 4): random Fourier features of x, y
        diff1,       # (T, 2): delta x, delta y
        direction,   # (T, 2): cos/sin of writing direction
        curvature,   # (T, 2): cos/sin of curvature
    ], axis=1)
    return features.astype(np.float32)

# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class AirHandwritingDataset(Dataset):
    def __init__(self, data_dir: str, label_path: str, language_path: str):
        self.tokens, self.token2idx, self.num_classes = load_language(language_path)
        self.word_labels = {}
        self.samples = []
        self.sample_words = []  # track which word each sample belongs to
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0: continue
                parts = line.split(",")
                origin, process = parts
                word = origin.strip()
                label_indices = encode_label(process.strip(), self.token2idx)
                self.word_labels[word] = label_indices

                folder_path = os.path.join(data_dir, word)
                for csv_file in sorted(os.listdir(folder_path)):
                    csv_path = os.path.join(folder_path, csv_file)
                    self.samples.append((csv_path, label_indices))
                    self.sample_words.append(word)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label_indices = self.samples[idx]
        df = pd.read_csv(csv_path)
        coord = df.to_numpy().astype(np.float32)
        features = extract_features(sample=coord)  # (T, 5)
        features = torch.from_numpy(features)
        label = torch.tensor(label_indices, dtype=torch.long)
        return features, label


def collate_fn(batch):
    features_list, labels_list = zip(*batch)

    input_lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    label_lengths = torch.tensor([l.size(0) for l in labels_list], dtype=torch.long)

    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_concat = torch.cat(labels_list)

    return features_padded, labels_concat, input_lengths, label_lengths

# ---------------------------------------------------------------------------
# 4. Model: Transformer + CTC
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Conformer Encoder Components
# ---------------------------------------------------------------------------

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """Feed-forward module used in Conformer blocks.
    Architecture: LayerNorm -> Linear(expand) -> Swish -> Dropout
                  -> Linear(project) -> Dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, T, d_model)
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise expansion (x2 for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        # Depthwise conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, T, d_model)
        x = self.layer_norm(x)
        # Transpose for Conv1d: (batch, d_model, T)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)   # (batch, 2*d_model, T)
        x = self.glu(x)               # (batch, d_model, T)
        x = self.depthwise_conv(x)     # (batch, d_model, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)    # (batch, d_model, T)
        x = self.dropout(x)
        # Transpose back: (batch, T, d_model)
        x= x.transpose(1, 2)
    
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x


class ConformerMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, T, d_model)
        x = self.layer_norm(x)
        attn_out, _ = self.mha(
            x, x, x, key_padding_mask=key_padding_mask,
        )
        return self.dropout(attn_out)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, d_ff, dropout)
        self.mhsa = ConformerMultiHeadSelfAttention(d_model, nhead, dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(d_model, d_ff, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, T, d_model)
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv_module(x, key_padding_mask=key_padding_mask)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks."""
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, d_ff, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class AirHandwritingTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = INPUT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        conv_kernel_size: int = 31,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Conformer encoder (replaces Transformer encoder)
        self.conformer_encoder = ConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            d_ff=dim_feedforward,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, src, src_lengths=None):
        x = self.input_proj(src)  # (batch, T, d_model)
        # Padding mask for sequence
        src_key_padding_mask = None
        if src_lengths is not None:
            batch_size, max_len, _ = x.size()
            src_key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= src_lengths.unsqueeze(1)

        x = self.pos_encoder(x)
        x = self.conformer_encoder(x, key_padding_mask=src_key_padding_mask)

        logits = self.output_proj(x)  # (batch, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = log_probs.permute(1, 0, 2)  # (T, batch, num_classes)

        return log_probs, src_lengths


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    label_path: str,
    language_path: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 128,
    nhead: int = 4,
    num_encoder_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "model.pth",
):
    dataset = AirHandwritingDataset(data_dir, label_path, language_path)
    tokens = dataset.tokens
    blank_idx = dataset.token2idx[BLANK]
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {dataset.num_classes} (including CTC blank)")

    # Stratified split: each word contributes 80% samples to train, 20% to val
    word_to_indices = {}
    for idx, word in enumerate(dataset.sample_words):
        word_to_indices.setdefault(word, []).append(idx)

    train_indices, val_indices = [], []
    for word, indices in word_to_indices.items():
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    print(f"Stratified split: {len(word_to_indices)} words, "
          f"{len(train_indices)} train / {len(val_indices)} val samples")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model = AirHandwritingTransformer(
        num_classes=dataset.num_classes,
        input_dim=INPUT_DIM,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    # CTC loss (blank=0)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for features, labels, input_lengths, label_lengths in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            log_probs, out_lengths = model(features, input_lengths)  # (T, batch, C)

            loss = ctc_loss_fn(log_probs, labels, out_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, input_lengths, label_lengths in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)

                log_probs, out_lengths = model(features, input_lengths)
                loss = ctc_loss_fn(log_probs, labels, out_lengths, label_lengths)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Print 3 sample predictions every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Grab first batch from val_loader
                sample_feats, sample_labels, sample_in_lens, sample_lab_lens = \
                    next(iter(val_loader))
                sample_feats = sample_feats.to(device)
                sample_in_lens = sample_in_lens.to(device)

                sample_log_probs, sample_out_lens = model(sample_feats, sample_in_lens)  # (T, B, C)

                # Split concatenated labels back per sample
                offset = 0
                n_show = min(3, sample_feats.size(0))
                for s in range(n_show):
                    lab_len = sample_lab_lens[s].item()
                    gt_indices = sample_labels[offset:offset + lab_len].tolist()
                    offset += lab_len

                    gt_str = "".join(tokens[idx] for idx in gt_indices)
                    pred_tokens = greedy_decode(
                        sample_log_probs[:sample_out_lens[s], s, :],
                        tokens, blank_idx,
                    )
                    pred_str = "".join(pred_tokens)
                    print(f"  [{s}] GT  : {gt_str}")
                    print(f"  [{s}] Pred: {pred_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# 6. Greedy CTC Decoding
# ---------------------------------------------------------------------------

def greedy_decode(log_probs, tokens, blank_idx=0):
    indices = log_probs.argmax(dim=-1).tolist()  # (T,)

    # Collapse repeats and remove blanks
    decoded_indices = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != blank_idx:
                decoded_indices.append(idx)
            prev = idx

    decoded = [tokens[i] for i in decoded_indices if 0 <= i < len(tokens)]
    return decoded


def beam_search_decode(log_probs, tokens, blank_idx=0, beam_width=5):
    
    # Convert to numpy if tensor
    if hasattr(log_probs, 'cpu'):
        log_probs = log_probs.cpu().numpy()
    
    T, num_classes = log_probs.shape
    NEG_INF = -float('inf')
    
    # Each beam: {prefix_tuple: (prob_blank, prob_non_blank)}
    beams = {(): (0.0, NEG_INF)}
    
    for t in range(T):
        new_beams = {}
        
        for prefix, (prob_b, prob_nb) in beams.items():
            prob_total = np.logaddexp(prob_b, prob_nb)
            
            for c in range(num_classes):
                p_c = log_probs[t, c]
                
                if c == blank_idx:
                    # Emit blank: prefix stays the same
                    new_prob_b = prob_total + p_c
                    if prefix in new_beams:
                        old_b, old_nb = new_beams[prefix]
                        new_beams[prefix] = (np.logaddexp(old_b, new_prob_b), old_nb)
                    else:
                        new_beams[prefix] = (new_prob_b, NEG_INF)
                else:
                    # Emit non-blank character
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same character as last: extend only if previous was blank
                        new_prefix = prefix + (c,)
                        new_prob_nb = prob_b + p_c
                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_b, np.logaddexp(old_nb, new_prob_nb))
                        else:
                            new_beams[new_prefix] = (NEG_INF, new_prob_nb)
                        
                        # Merge (repeat same char without blank)
                        merge_prob_nb = prob_nb + p_c
                        if prefix in new_beams:
                            old_b, old_nb = new_beams[prefix]
                            new_beams[prefix] = (old_b, np.logaddexp(old_nb, merge_prob_nb))
                        else:
                            new_beams[prefix] = (NEG_INF, merge_prob_nb)
                    else:
                        # Different character: extend prefix
                        new_prefix = prefix + (c,)
                        new_prob_nb = prob_total + p_c
                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_b, np.logaddexp(old_nb, new_prob_nb))
                        else:
                            new_beams[new_prefix] = (NEG_INF, new_prob_nb)
        
        # Prune to beam_width
        beam_list = [(prefix, np.logaddexp(pb, pnb)) 
                     for prefix, (pb, pnb) in new_beams.items()]
        beam_list.sort(key=lambda x: x[1], reverse=True)
        beam_list = beam_list[:beam_width]
        
        beams = {prefix: new_beams[prefix] for prefix, _ in beam_list}
    
    # Get best beam
    best_prefix = max(beams.keys(), 
                      key=lambda p: np.logaddexp(beams[p][0], beams[p][1]))
    
    decoded = [tokens[i] for i in best_prefix if 0 <= i < len(tokens)]
    return decoded


# ---------------------------------------------------------------------------
# 7. Inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, language_path: str, device: str = None,
               d_model=128, nhead=4, num_encoder_layers=4,
               dim_feedforward=256):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens, token2idx, num_classes = load_language(language_path)
    blank_idx = token2idx[BLANK]

    model = AirHandwritingTransformer(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokens, blank_idx, device


def predict_from_coordinates(model, coord: np.ndarray, tokens, blank_idx, device):
    coord = normalize_coordinates(coord.astype(np.float32))
    features = extract_features(coord)
    features_t = torch.from_numpy(features).unsqueeze(0).to(device)
    length = torch.tensor([features_t.size(1)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        log_probs, _ = model(features_t, length)

    # decoded_tokens = greedy_decode(log_probs[:, 0, :], tokens, blank_idx)
    decoded_tokens = beam_search_decode(log_probs[:, 0, :], tokens, blank_idx)
    raw_string = "".join(decoded_tokens)
    return decoded_tokens, raw_string