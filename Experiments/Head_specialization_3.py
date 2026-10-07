"""
W&B Report Section 3. Attention Rollout & Head Specialization

Extracts attention weights from the LAST encoder layer for a single
German→English sentence, then logs one heatmap per head to W&B.

"""

import argparse
import math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # headless backend no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb

from model import Transformer, make_src_mask, MultiHeadAttention
from dataset import Multi30kDataset


# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint_best_noam_scheduler.pth",
    help="Path to the saved .pth checkpoint from experiment 2.1",
)
parser.add_argument(
    "--d_model",   type=int,   default=256,
    help="Must match the checkpoint's d_model",
)
parser.add_argument(
    "--n_layers",  type=int,   default=3,
    help="Must match the checkpoint's N",
)
parser.add_argument(
    "--num_heads", type=int,   default=8,
    help="Must match the checkpoint's num_heads",
)
parser.add_argument(
    "--d_ff",      type=int,   default=512,
    help="Must match the checkpoint's d_ff",
)
args = parser.parse_args()


# Sentence to visualise 
# Pick a short, interesting German sentence from Multi30k validation set.
# Short sentences make cleaner heatmaps.

SENTENCE_DE = "Ein Mann sitzt auf einer Bank im Park ."
# Ground-truth English: "A man is sitting on a bench in the park ."


# Helpers

def load_model_from_checkpoint(path: str, d_model, n_layers, num_heads, d_ff, device):
    """Load Transformer and return (model, src_vocab, tgt_vocab)."""
    checkpoint = torch.load(path, map_location=device)

    model = Transformer(
        d_model=d_model,
        N=n_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,   # no dropout during inference
    ).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from '{path}'")
    return model


def tokenize_src(sentence: str, model: Transformer):
    """
    Tokenise a German sentence and convert to a [1, seq_len] index tensor.
    Wraps with <sos> and <eos>.
    """
    tokens = [tok.text for tok in model.src_tokenizer(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]

    w2i    = model.src_vocab["word2idx"]
    unk    = model.src_vocab["unk_idx"]
    ids    = [w2i.get(t, unk) for t in tokens]

    return tokens, torch.tensor([ids], dtype=torch.long)


# Attention hook 

class AttentionCaptureHook:
    """
    Registers a forward hook on a MultiHeadAttention module.

    Because MHA.forward returns only the output (attn_weights are
    discarded), we hook into scaled_dot_product_attention by temporarily
    replacing the module-level function with a wrapper that saves the
    weights as a side-effect.

    The hook is installed on the MHA module's forward call.
    We recompute attention weights from Q and K projections directly.
    """

    def __init__(self, mha_module: MultiHeadAttention):
        self.mha   = mha_module
        self.weights = None          # will hold [num_heads, seq, seq] after capture
        self._hook  = mha_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        """
        inputs = (query, key, value, mask) passed to MHA.forward
        We re-run just the Q/K projection + scaled dot-product to get weights.
        """
        query, key, value = inputs[0], inputs[1], inputs[2]
        mask = inputs[3] if len(inputs) > 3 else None

        with torch.no_grad():
            batch_size = query.size(0)
            d_k        = module.d_k
            num_heads  = module.num_heads

            Q = module.W_Q(query).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
            K = module.W_K(key  ).view(batch_size, -1, num_heads, d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))

            attn_w = F.softmax(scores, dim=-1)   # [batch, heads, seq_q, seq_k]

        # store for the single sentence (batch=1)
        self.weights = attn_w[0].cpu()           # [num_heads, seq_q, seq_k]

    def remove(self):
        self._hook.remove()


# Heatmap drawing

def draw_heatmap(attn: np.ndarray, tokens: list[str], head_idx: int) -> plt.Figure:
    """
    Draw a single attention heatmap for one head.

    attn   : numpy array [seq, seq]
    tokens : list of token strings (both axes)
    """
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.55),
                                    max(5, len(tokens) * 0.5)))

    im = ax.imshow(attn, cmap="Blues", vmin=0.0, vmax=attn.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)

    ax.set_xlabel("Key (attended to)",  fontsize=10)
    ax.set_ylabel("Query (attends from)", fontsize=10)
    ax.set_title(f"Encoder Last Layer - Head {head_idx}", fontsize=11, fontweight="bold")

    # Annotate cells with weight values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attn[i, j]
            color = "white" if val > 0.5 * attn.max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.tight_layout()
    return fig


def draw_all_heads_grid(all_weights: list[np.ndarray], tokens: list[str]) -> plt.Figure:
    """
    Draw all heads in a single grid figure (2 rows × 4 cols for 8 heads).
    Logged as one summary image to W&B.
    """
    num_heads = len(all_weights)
    ncols = 4
    nrows = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * max(4, len(tokens) * 0.45),
                 nrows * max(3.5, len(tokens) * 0.4))
    )
    axes = axes.flatten()

    for h, (attn, ax) in enumerate(zip(all_weights, axes)):
        im = ax.imshow(attn, cmap="Blues", vmin=0.0, vmax=attn.max())
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Head {h}", fontsize=9, fontweight="bold")

    # hide unused subplots
    for ax in axes[num_heads:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Encoder Last Layer - All Heads\n\"{SENTENCE_DE}\"",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


# Head similarity (for Head Redundancy analysis) 

def compute_head_similarity(all_weights: list[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity between flattened attention maps.
    Returns a [num_heads, num_heads] similarity matrix.
    """
    flat = np.array([w.flatten() for w in all_weights])   # [H, seq*seq]
    # cosine similarity
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normed = flat / norms
    return normed @ normed.T                               # [H, H]


def draw_similarity_matrix(sim: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax)
    n = sim.shape[0]
    ax.set_xticks(range(n)); ax.set_xticklabels([f"H{i}" for i in range(n)])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"H{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Head Similarity Matrix\n(cosine sim of attention maps)", fontsize=10)
    fig.tight_layout()
    return fig


# Main 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # W&B 
    wandb.init(
        project="da6401_assignment3",
        name="attention_head_specialization",
        group="experiment_2_3",
        config={
            "sentence": SENTENCE_DE,
            "checkpoint": args.checkpoint,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
        },
    )

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
    )

    # Tokenise sentence
    tokens, src_tensor = tokenize_src(SENTENCE_DE, model)
    src_tensor = src_tensor.to(device)
    src_mask   = make_src_mask(src_tensor, model.src_pad_idx)

    print(f"\nTokens ({len(tokens)}): {tokens}")

    # Install hook on last encoder layer's self_attn 
    last_layer = model.encoder.layers[-1]
    hook       = AttentionCaptureHook(last_layer.self_attn)

    # Run encoder (triggers the hook) 
    with torch.no_grad():
        _ = model.encode(src_tensor, src_mask)

    hook.remove()   # clean up immediately

    # hook.weights shape: [num_heads, seq_len, seq_len]
    attn_weights = hook.weights.numpy()   # [H, seq, seq]
    num_heads    = attn_weights.shape[0]

    print(f"Captured attention weights: {attn_weights.shape}")

    # Per-head heatmaps
    head_images = {}
    all_maps    = []

    for h in range(num_heads):
        head_map = attn_weights[h]          # [seq, seq]
        all_maps.append(head_map)

        fig = draw_heatmap(head_map, tokens, head_idx=h)
        head_images[f"head_{h:02d}_heatmap"] = wandb.Image(
            fig,
            caption=f"Head {h} | last encoder layer | \"{SENTENCE_DE}\""
        )
        plt.close(fig)

    # All-heads grid 
    grid_fig = draw_all_heads_grid(all_maps, tokens)
    head_images["all_heads_grid"] = wandb.Image(
        grid_fig,
        caption=f"All {num_heads} heads | last encoder layer"
    )
    plt.close(grid_fig)

    # Head similarity matrix 
    sim_matrix = compute_head_similarity(all_maps)
    sim_fig    = draw_similarity_matrix(sim_matrix)
    head_images["head_similarity_matrix"] = wandb.Image(
        sim_fig,
        caption="Cosine similarity between flattened attention maps (head redundancy)"
    )
    plt.close(sim_fig)

    # Entropy per head (lower = more focused / specialised) 
    # Entropy of the attention distribution for each head, averaged over positions
    entropies = []
    for h in range(num_heads):
        p    = attn_weights[h] + 1e-9          # avoid log(0)
        ent  = -np.sum(p * np.log(p), axis=-1) # [seq]
        entropies.append(float(ent.mean()))

    entropy_fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(range(num_heads), entropies, color="steelblue", edgecolor="black")
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"Head {h}" for h in range(num_heads)], rotation=30, ha="right")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title("Per-Head Attention Entropy\n(lower = more specialised / focused)")
    for bar, val in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    entropy_fig.tight_layout()
    head_images["head_entropy"] = wandb.Image(
        entropy_fig,
        caption="Attention entropy per head - lower means more focused attention pattern"
    )
    plt.close(entropy_fig)

    # Log everything at once
    wandb.log(head_images)

    #  Print summary to console 
    print("\n Head Analysis Summary")
    print(f"{'Head':<6} {'Mean Entropy':<15} {'Max Attn Token Pair'}")
    print("-" * 50)
    for h in range(num_heads):
        m         = all_maps[h]
        i, j      = np.unravel_index(m.argmax(), m.shape)
        pair      = f"{tokens[i]!r:12s} → {tokens[j]!r}"
        print(f"  {h:<4} {entropies[h]:<15.4f} {pair}")

    print("\n Head Similarity (pairs with cosine sim > 0.85 may be redundant) ")
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            if sim_matrix[i, j] > 0.85:
                print(f"  Head {i} ↔ Head {j}  similarity={sim_matrix[i,j]:.3f}  ← REDUNDANT")

    print(f"\nAll plots logged to W&B run: {wandb.run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
