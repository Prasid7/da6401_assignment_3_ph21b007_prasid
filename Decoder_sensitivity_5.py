"""
W&B Report Section 5. Decoder Sensitivity: Label Smoothing (ε=0.1 vs ε=0.0)

Runs two training experiments back-to-back:
  Run A: Label smoothing ε = 0.1  (standard Transformer setting)
  Run B: Label smoothing ε = 0.0  (vanilla cross-entropy)

Both runs use identical model architecture, dataset, seed, and optimizer.
Logs per epoch to W&B:
  - train_loss
  - val_loss
  - val_bleu          (every BLEU_EVERY epochs)
  - pred_confidence   (mean softmax probability of the correct token on val set)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from train import run_epoch, evaluate_bleu, save_checkpoint, LabelSmoothingLoss


# Shared hyperparameters
SEED         = 42
BATCH_SIZE   = 128
NUM_EPOCHS   = 30
D_MODEL      = 256
N_LAYERS     = 3
NUM_HEADS    = 8
D_FF         = 512
DROPOUT      = 0.1
WARMUP_STEPS = 4000
BLEU_EVERY   = 2     # evaluate BLEU every N epochs (slow — greedy decodes full val set)


# Reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Prediction Confidence metric

@torch.no_grad()
def evaluate_confidence(
    model:      Transformer,
    val_loader: DataLoader,
    tgt_pad_idx: int,
    device:     torch.device,
) -> float:
    """
    Compute mean softmax probability assigned to the correct token
    across all non-pad positions in the validation set.

    High confidence (→ 1.0) means the model is very "sure" about its
    predictions. With ε=0.0 this will climb much higher than ε=0.1,
    illustrating the over-confidence / poor-calibration problem.

    Args:
        model        : Trained Transformer (in eval mode).
        val_loader   : Validation DataLoader.
        tgt_pad_idx  : Index of <pad> in the target vocabulary.
        device       : Torch device.

    Returns:
        float: Mean confidence over all non-pad token positions.
    """
    model.eval()

    total_confidence = 0.0
    total_tokens     = 0

    for src, tgt in val_loader:
        src = src.to(device)   # [batch, src_len]
        tgt = tgt.to(device)   # [batch, tgt_len]

        tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
        tgt_output = tgt[:, 1:]    # [batch, tgt_len-1]  — gold labels

        src_mask = make_src_mask(src, model.src_pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, model.tgt_pad_idx)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        # logits: [batch, tgt_len-1, vocab_size]

        # Softmax → probabilities
        probs = F.softmax(logits, dim=-1)   # [batch, tgt_len-1, vocab_size]

        # Gather the probability of the correct (gold) token at each position
        # tgt_output: [batch, tgt_len-1]  → unsqueeze for gather
        gold_probs = probs.gather(
            dim=-1,
            index=tgt_output.unsqueeze(-1),   # [batch, tgt_len-1, 1]
        ).squeeze(-1)                          # [batch, tgt_len-1]

        # Mask out pad positions
        non_pad = (tgt_output != tgt_pad_idx)   # [batch, tgt_len-1]

        total_confidence += gold_probs[non_pad].sum().item()
        total_tokens     += non_pad.sum().item()

    return total_confidence / max(total_tokens, 1)


# Dataset

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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad, tgt_pad_idx=tgt_pad),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=partial(collate_fn, src_pad_idx=src_pad, tgt_pad_idx=tgt_pad),
    )
    return train_loader, val_loader, train_dataset, val_dataset


# Single experiment runner

def run_experiment(
    run_name:      str,
    smoothing:     float,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    train_dataset: Multi30kDataset,
    val_dataset:   Multi30kDataset,
    device:        torch.device,
):
    """
    Train one model with a fixed label-smoothing value and log all
    metrics (train_loss, val_loss, val_bleu, pred_confidence) to W&B.

    Args:
        run_name  : W&B run display name.
        smoothing : ε for LabelSmoothingLoss — 0.1 or 0.0.
    """

    config = {
        "smoothing":     smoothing,
        "d_model":       D_MODEL,
        "N":             N_LAYERS,
        "num_heads":     NUM_HEADS,
        "d_ff":          D_FF,
        "dropout":       DROPOUT,
        "warmup_steps":  WARMUP_STEPS,
        "batch_size":    BATCH_SIZE,
        "num_epochs":    NUM_EPOCHS,
        "seed":          SEED,
    }

    wandb.init(
        project="da6401_assignment3",
        group="experiment_2_5_label_smoothing",
        name=run_name,
        config=config,
        reinit=True,
    )

    set_seed(SEED)

    # Build model 
    model = Transformer(
        d_model=D_MODEL,
        N=N_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    tgt_vocab_size = train_dataset.tgt_vocab['vocab_size']
    tgt_pad_idx    = train_dataset.tgt_vocab['pad_idx']

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{run_name}] Total params: {total_params:,}")
    wandb.config.update({"total_params": total_params})

    # Loss / optimizer / scheduler 
    # The ONLY difference between the two runs is the smoothing value here.
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=tgt_pad_idx,
        smoothing=smoothing,
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

    # Training loop 
    best_val_bleu = 0.0

    for epoch in range(NUM_EPOCHS):

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )

        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch, is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]['lr']

        # Prediction confidence — computed every epoch (cheap: single forward pass)
        pred_confidence = evaluate_confidence(
            model, val_loader, tgt_pad_idx, device=device
        )

        # BLEU — computed every BLEU_EVERY epochs (slow: greedy decoding)
        val_bleu = None
        if epoch % BLEU_EVERY == 0:
            val_bleu = evaluate_bleu(
                model, val_loader, val_dataset.tgt_vocab, device=device
            )
            print(f"  [BLEU] val_bleu = {val_bleu:.2f}")

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                ckpt_path = f"checkpoint_best_{run_name.replace(' ', '_')}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, path=ckpt_path)
                print(f"  [CKPT] Saved → {ckpt_path}")

        log_dict = {
            "epoch":            epoch,
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "lr":               current_lr,
            "pred_confidence":  pred_confidence,
        }
        if val_bleu is not None:
            log_dict["val_bleu"] = val_bleu

        wandb.log(log_dict)

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | confidence={pred_confidence:.4f} | "
            f"lr={current_lr:.2e}"
        )

    print(f"\n[{run_name}] Best val BLEU = {best_val_bleu:.2f}\n")
    wandb.finish()


# Entry point 

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    # Run A: Label Smoothing ε = 0.1
    print("\n" + "="*60)
    print("RUN A: Label Smoothing  ε = 0.1")
    print("="*60)
    run_experiment(
        run_name="label_smoothing_0.1",
        smoothing=0.1,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    # Run B: No Label Smoothing (ε = 0.0 → vanilla cross-entropy) 
    print("\n" + "="*60)
    print("RUN B: No Label Smoothing  ε = 0.0  (cross-entropy)")
    print("="*60)
    run_experiment(
        run_name="label_smoothing_0.0",
        smoothing=0.0,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    print("\nBoth runs complete.")
