# DA6401 Assignment 3 - Transformer for German→English Machine Translation

**Student:** Prasid | **Roll No:** PH21B007

🔗 [W&B Report](https://wandb.ai/prasid-indian-institute-of-technology-madras/da6401_assignment3/reports/DA6401-Assignment-3-PH21B007-PRASID--VmlldzoxNjkzMDIyMw) &nbsp;|&nbsp; 🔗 [GitHub Repository](https://github.com/Prasid7/da6401_assignment_3_ph21b007_prasid)

---


Implementation of the **"Attention Is All You Need"** (Vaswani et al., 2017) Transformer architecture from scratch in PyTorch, applied to Neural Machine Translation on the Multi30k dataset.

---

## Project Structure

```
da6401_assignment_3/
├── model.py          # Full Transformer architecture
├── train.py          # Training pipeline, loss, BLEU evaluation, checkpointing
├── dataset.py        # Multi30k dataset class, vocab building, collate function
├── lr_scheduler.py   # Noam learning rate scheduler
└── README.md
```

---

## Architecture Overview

The model follows the original paper exactly — a 6-layer encoder-decoder Transformer with multi-head self-attention, cross-attention, and position-wise feed-forward sub-layers.

### `model.py`

**`scaled_dot_product_attention(Q, K, V, mask)`**  
Implements the core attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Returns both the attended output and the attention weight matrix. Mask positions are filled with `-inf` before the softmax, driving their weight to zero.

**`make_src_mask(src, pad_idx)`**  
Builds a padding mask of shape `[batch, 1, 1, src_len]` for the encoder. `True` where the token is `<pad>`.

**`make_tgt_mask(tgt, pad_idx)`**  
Builds a combined padding + causal (look-ahead) mask of shape `[batch, 1, tgt_len, tgt_len]` for the decoder. Uses `torch.triu` to prevent any position from attending to future tokens.

**`MultiHeadAttention`**  
Projects Q, K, V into `num_heads` subspaces (each of depth `d_k = d_model // num_heads`), applies scaled dot-product attention in parallel, concatenates the results, and projects back to `d_model`. `torch.nn.MultiheadAttention` is **not** used.

**`PositionalEncoding`**  
Sinusoidal encoding following:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Pre-computed up to `max_len=5000` and registered as a non-trainable buffer. Embeddings are scaled by $\sqrt{d_{model}}$ before the encoding is added.

**`PositionwiseFeedForward`**  
Two-layer linear transformation: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`, with dropout between the two projections.

**`EncoderLayer` / `DecoderLayer`**  
Both follow the **Post-LayerNorm** (original paper) convention — residual connection first, then LayerNorm. The decoder layer has three sub-layers: masked self-attention, cross-attention over encoder memory, and FFN.

**`Encoder` / `Decoder`**  
Stacks of `N` deep-copied encoder/decoder layers with a final `nn.LayerNorm`.

**`Transformer`**  
The full model. On construction, it auto-downloads spaCy models (`de_core_news_sm`, `en_core_web_sm`), builds vocabularies from the Multi30k training split, and attempts to load a pre-trained checkpoint from Google Drive (gracefully skips if unavailable). Exposes `encode`, `decode`, `forward`, and `infer` methods.

`infer(src_sentence: str) -> str` runs greedy autoregressive decoding with punctuation re-attachment for clean output.

---

### `lr_scheduler.py`

**`NoamScheduler`**  
Subclasses `torch.optim.lr_scheduler.LRScheduler`. Implements the Noam schedule:

$$lrate = d_{model}^{-0.5} \cdot \min\left(step-num^{-0.5},\ step-num \cdot warmup-steps^{-1.5}\right)$$

LR increases linearly during warm-up and then decays inversely proportional to the square root of the step number. The scale multiplies the optimizer's base learning rate (set to `1.0` at init, so the scheduler has full control).

**`get_lr_history`** - a helper to simulate the LR trajectory for visualization/debugging without touching real training.

---

### `dataset.py`

**`Multi30kDataset`**  
A `torch.utils.data.Dataset` wrapping the [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset. On first construction it:
1. Loads the Hugging Face dataset for the requested split.
2. Tokenizes with spaCy (`de_core_news_sm` for German, `en_core_web_sm` for English).
3. Builds word→index and index→word mappings (vocabulary is always built from the training split, never validation/test).
4. Converts all sentences to integer index lists padded with `<sos>` and `<eos>`.

Validation and test datasets accept pre-built `src_vocab` / `tgt_vocab` to ensure consistent indexing.

**`collate_fn`**  
Pads variable-length sequences within a batch to the maximum sequence length in that batch, keeping padding to a minimum.

---

### `train.py`

**`LabelSmoothingLoss`**  
Implements label smoothing (ϵ = 0.1). The smoothed target distribution assigns `1 - ϵ` to the correct token and ` ϵ / (vocab_size - 2)` to all other tokens (excluding `<pad>`). Rows corresponding to `<pad>` targets are zeroed out and excluded from the mean.

**`run_epoch`**  
Handles both training and evaluation in one function controlled by the `is_train` flag. Includes gradient clipping at `max_norm = 1.0`.

**`greedy_decode`**  
Autoregressive token-by-token generation. Encodes the source once, then repeatedly runs the decoder on the growing output sequence until `<eos>` or `max_len` is reached.

**`evaluate_bleu`**  
Computes corpus-level BLEU. Uses `sacrebleu` (preferred, matches Gradescope evaluation) when available, falling back to `nltk.translate.bleu_score.corpus_bleu` otherwise.

**`save_checkpoint` / `load_checkpoint`**  
Save/restore full training state: model weights, optimizer state, scheduler state, epoch number, model config, and vocabularies.

**`run_training_experiment`**  
The main entry point. Configures a full training run with W&B logging, evaluating validation BLEU every 2 epochs and saving the best checkpoint.

---

## Setup

```bash
# Install dependencies
pip install torch torchvision datasets spacy wandb sacrebleu nltk tqdm gdown

# Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
python train.py
```

Default hyperparameters (configurable in `run_training_experiment`):

| Hyperparameter  | Value  |
|-----------------|--------|
| `d_model`       | 512    |
| `N` (layers)    | 6      |
| `num_heads`     | 8      |
| `d_ff`          | 2048   |
| `dropout`       | 0.3    |
| `batch_size`    | 128    |
| `num_epochs`    | 50     |
| `warmup_steps`  | 4000   |
| `label_smooth`  | 0.1    |

The best checkpoint is saved to `checkpoint_best.pth` whenever validation BLEU improves.

---

## Inference

```python
from model import Transformer

model = Transformer()          # loads vocab and checkpoint automatically
print(model.infer("Ein Mann sitzt auf einer Bank."))
# → "A man is sitting on a bench."
```

---

## Design Notes

**Post-LayerNorm vs Pre-LayerNorm**  
This implementation uses **Post-LayerNorm** (residual → LayerNorm), matching the original paper. While Pre-LayerNorm is often more training-stable, Post-LayerNorm combined with the Noam scheduler's warmup phase achieves good convergence on Multi30k and faithfully reproduces the paper's design.

**Vocabulary**  
Vocabularies are built from the training split only and shared across validation/test to prevent data leakage.

**Scaling**  
Source and target token embeddings are multiplied by $\sqrt{d_{model}}$ before positional encoding is added, as specified in §3.4 of the paper.

---

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). *NeurIPS 2017*.

Dataset: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
