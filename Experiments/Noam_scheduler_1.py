"""
W&B Report Section 1. Noam Scheduler vs. Fixed Learning Rate

Runs two training experiments back-to-back:
  Run A: Noam LR scheduler  (warmup_steps=4000)
  Run B: Fixed LR = 1e-4    (no scheduler)

Both runs use identical model architecture, dataset, and random seed.
Logs train_loss, val_loss, val_bleu, and lr at every epoch to W&B.
"""

import torch
import torch.nn as nn
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
NUM_EPOCHS   = 30       # 30 epochs is enough to clearly see divergence vs. convergence
D_MODEL      = 256      # smaller model so both runs finish faster
N_LAYERS     = 3
NUM_HEADS    = 8
D_FF         = 512
DROPOUT      = 0.1
WARMUP_STEPS = 4000
FIXED_LR     = 1e-4
BLEU_EVERY   = 2        # evaluate BLEU every N epochs (slow — runs infer() on 1014 sentences)


# Reproducibility 
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Build datasets once reused for both runs 
def build_dataloaders():
    print("Loading datasets...")
    train_dataset = Multi30kDataset(split='train')
    val_dataset   = Multi30kDataset(split='validation',
                                    src_vocab=train_dataset.src_vocab,
                                    tgt_vocab=train_dataset.tgt_vocab)

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


# Single experiment runner 
def run_experiment(
    run_name: str,
    use_noam: bool,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    train_dataset: Multi30kDataset,
    val_dataset:   Multi30kDataset,
    device: torch.device,
):
    """
    Initialises a fresh Transformer and trains it for NUM_EPOCHS.

    Args:
        run_name   : W&B run display name.
        use_noam   : True → Noam scheduler; False → fixed LR (no scheduler).
        *_loader   : Pre-built DataLoaders.
        *_dataset  : Datasets (needed for vocab info and BLEU eval).
        device     : torch.device.
    """

    config = {
        "scheduler":   "noam" if use_noam else "fixed_lr",
        "lr":          1.0 if use_noam else FIXED_LR,   # base LR; Noam scales it
        "warmup_steps": WARMUP_STEPS if use_noam else None,
        "batch_size":  BATCH_SIZE,
        "num_epochs":  NUM_EPOCHS,
        "d_model":     D_MODEL,
        "N":           N_LAYERS,
        "num_heads":   NUM_HEADS,
        "d_ff":        D_FF,
        "dropout":     DROPOUT,
        "seed":        SEED,
    }

    # init W&B run 
    wandb.init(
        project="da6401_assignment3",
        group="experiment_2_1_noam_vs_fixed",
        name=run_name,
        config=config,
        reinit=True,
    )

    set_seed(SEED)  # reset seed so both runs start from identical weights

    # model
    model = Transformer(
        d_model=D_MODEL,
        N=N_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    tgt_vocab_size = train_dataset.tgt_vocab['vocab_size']
    tgt_pad_idx    = train_dataset.tgt_vocab['pad_idx']

    # loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=tgt_pad_idx,
        smoothing=0.1,
    ).to(device)

    # optimizer 
    # Both runs use Adam with the paper's beta/eps.
    # For Noam: lr=1.0 as base (Noam formula multiplies this).
    # For Fixed: lr=FIXED_LR with no scheduler.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0 if use_noam else FIXED_LR,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # scheduler 
    scheduler = NoamScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS) \
                if use_noam else None

    # training loop
    best_val_bleu = 0.0

    for epoch in range(NUM_EPOCHS):

        # train 
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        # validate
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        current_lr = optimizer.param_groups[0]['lr']

        # BLEU (every BLEU_EVERY epochs)
        val_bleu = None
        if epoch % BLEU_EVERY == 0:
            val_bleu = evaluate_bleu(model, val_loader, val_dataset.tgt_vocab, device=device)
            print(f"  [BLEU] val_bleu = {val_bleu:.2f}")

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                ckpt_path = f"checkpoint_best_{run_name.replace(' ', '_')}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, path=ckpt_path)
                print(f"  [CKPT] New best checkpoint saved → {ckpt_path}")

        # log to W&B
        log_dict = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        }
        if val_bleu is not None:
            log_dict["val_bleu"] = val_bleu

        wandb.log(log_dict)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

    print(f"\n[{run_name}] Best val BLEU = {best_val_bleu:.2f}\n")
    wandb.finish()


# Entry point 
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets once shared between both runs to save time
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    # Run A: Noam Scheduler 
    print("\n" + "="*60)
    print("RUN A: Noam LR Scheduler")
    print("="*60)
    run_experiment(
        run_name="noam_scheduler",
        use_noam=True,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    # Run B: Fixed LR 
    print("\n" + "="*60)
    print("RUN B: Fixed LR = 1e-4")
    print("="*60)
    run_experiment(
        run_name="fixed_lr_1e-4",
        use_noam=False,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    print("\nBoth runs complete.")
