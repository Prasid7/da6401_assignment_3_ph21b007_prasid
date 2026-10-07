"""
W&B Report Section 2. Ablation: With vs. Without the √(1/dₖ) Scaling Factor

Runs two training experiments back-to-back:
  Run A: Scaled attention   → scores = Q·Kᵀ / √dₖ   (original)
  Run B: Unscaled attention → scores = Q·Kᵀ           (ablation)

Key metric logged every step for the first 1000 training steps:
  - wq_grad_norm : gradient norm of W_Q.weight in the first encoder layer
  - wk_grad_norm : gradient norm of W_K.weight in the first encoder layer
  - train_loss

After 1000 steps the script also logs val_loss so we can see the
downstream translation quality gap caused by the vanishing gradients.

"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from typing import Optional, Tuple
import wandb

import model as model_module                          # so we can monkey-patch
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, save_checkpoint


# Shared hyperparameters 
SEED          = 42
BATCH_SIZE    = 128
MAX_STEPS     = 1000   # paper asks for first 1000 steps for grad norm analysis
D_MODEL       = 256    # smaller = faster; use 512 if you have time/GPU
N_LAYERS      = 3
NUM_HEADS     = 8
D_FF          = 512
DROPOUT       = 0.1
WARMUP_STEPS  = 4000


# Reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Replacement attention functions 

def scaled_dot_product_attention_WITH_scale(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original: divides by √dₖ (correct implementation)."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def scaled_dot_product_attention_NO_scale(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ablation: NO division by √dₖ — raw dot products."""
    scores = torch.matmul(Q, K.transpose(-2, -1))   # ← only change
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# Dataset (built once, shared) 
def build_dataloaders():
    print("Loading datasets...")
    train_dataset = Multi30kDataset(split='train')
    val_dataset   = Multi30kDataset(
        split='validation',
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    src_pad = train_dataset.src_vocab['pad_idx']
    tgt_pad = train_dataset.tgt_vocab['pad_idx']

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad, tgt_pad_idx=tgt_pad),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad, tgt_pad_idx=tgt_pad),
    )
    return train_loader, val_loader, train_dataset, val_dataset


# Core training loop (step-level, not epoch-level)
def run_steps(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    model:        Transformer,
    loss_fn:      torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler:    NoamScheduler,
    max_steps:    int,
    device:       torch.device,
):
    """
    Trains for `max_steps` gradient steps (not epochs).
    Logs at every step:
      - train_loss
      - wq_grad_norm  (W_Q of encoder layer 0, self-attention)
      - wk_grad_norm  (W_K of encoder layer 0, self-attention)
      - lr

    Returns average val_loss evaluated after training completes.
    """
    model.train()

    # Grab references to the W_Q / W_K in the FIRST encoder layer's
    # self-attention head — this is the most informative layer to watch.
    enc_layer_0_attn = model.encoder.layers[0].self_attn
    W_Q = enc_layer_0_attn.W_Q
    W_K = enc_layer_0_attn.W_K

    step = 0
    data_iter = iter(train_loader)

    while step < max_steps:
        # Restart dataloader if we exhaust it before hitting max_steps
        try:
            src, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            src, tgt = next(data_iter)

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, model.src_pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, model.tgt_pad_idx)

        # forward
        logits      = model(src, tgt_input, src_mask, tgt_mask)
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_output.reshape(-1)
        loss        = loss_fn(logits_flat, target_flat)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # capture grad norms BEFORE optimizer.step() 
        wq_grad_norm = W_Q.weight.grad.norm().item() if W_Q.weight.grad is not None else 0.0
        wk_grad_norm = W_K.weight.grad.norm().item() if W_K.weight.grad is not None else 0.0

        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # log to W&B 
        wandb.log({
            "step":          step,
            "train_loss":    loss.item(),
            "wq_grad_norm":  wq_grad_norm,
            "wk_grad_norm":  wk_grad_norm,
            "lr":            current_lr,
        })

        if step % 100 == 0:
            print(
                f"  step={step:4d} | loss={loss.item():.4f} | "
                f"wq_grad={wq_grad_norm:.4e} | wk_grad={wk_grad_norm:.4e} | "
                f"lr={current_lr:.2e}"
            )

        step += 1

    # validation loss after training
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src, model.src_pad_idx)
            tgt_mask   = make_tgt_mask(tgt_input, model.tgt_pad_idx)
            logits      = model(src, tgt_input, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_output.reshape(-1)
            loss        = loss_fn(logits_flat, target_flat)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    wandb.log({"val_loss_after_training": avg_val_loss})
    return avg_val_loss


# Single experiment runner
def run_experiment(
    run_name:      str,
    use_scaling:   bool,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    train_dataset: Multi30kDataset,
    val_dataset:   Multi30kDataset,
    device:        torch.device,
):
    """
    Args:
        use_scaling : True  → original √dₖ scaling (Run A)
                      False → no scaling, raw dot products (Run B)
    """

    # monkey-patch the module-level function BEFORE model is built 
    # MultiHeadAttention.forward calls model_module.scaled_dot_product_attention
    # via direct name reference, so patching the module attribute is enough.
    if use_scaling:
        model_module.scaled_dot_product_attention = scaled_dot_product_attention_WITH_scale
    else:
        model_module.scaled_dot_product_attention = scaled_dot_product_attention_NO_scale

    config = {
        "scaling":     "with_sqrt_dk" if use_scaling else "no_scaling",
        "d_k":         D_MODEL // NUM_HEADS,   # 32 for d_model=256, h=8
        "max_steps":   MAX_STEPS,
        "batch_size":  BATCH_SIZE,
        "d_model":     D_MODEL,
        "N":           N_LAYERS,
        "num_heads":   NUM_HEADS,
        "d_ff":        D_FF,
        "dropout":     DROPOUT,
        "warmup_steps": WARMUP_STEPS,
        "seed":        SEED,
    }

    wandb.init(
        project="da6401_assignment3",
        group="experiment_2_2_scaling_ablation",
        name=run_name,
        config=config,
        reinit=True,
    )

    set_seed(SEED)

    # build the model
    model = Transformer(
        d_model=D_MODEL,
        N=N_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    tgt_vocab_size = train_dataset.tgt_vocab['vocab_size']
    tgt_pad_idx    = train_dataset.tgt_vocab['pad_idx']

    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=tgt_pad_idx,
        smoothing=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
    )

    # train for MAX_STEPS steps
    print(f"\n{'='*60}")
    print(f"  {run_name}  |  d_k={D_MODEL // NUM_HEADS}  |  scaling={use_scaling}")
    print(f"{'='*60}")

    avg_val_loss = run_steps(
        train_loader, val_loader,
        model, loss_fn, optimizer, scheduler,
        max_steps=MAX_STEPS,
        device=device,
    )

    print(f"\n[{run_name}] val_loss after {MAX_STEPS} steps = {avg_val_loss:.4f}")
    wandb.finish()

    # restore original function so Run A's patch doesn't bleed into Run B 
    model_module.scaled_dot_product_attention = scaled_dot_product_attention_WITH_scale


# Entry point
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    # Run A: WITH √dₖ scaling 
    run_experiment(
        run_name="with_sqrt_dk_scaling",
        use_scaling=True,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    # Run B: NO scaling
    run_experiment(
        run_name="no_scaling",
        use_scaling=False,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    print("\nBoth runs complete.")
    
