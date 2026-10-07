"""
W&B Report Section 4. Positional Encoding vs. Learned Positional Embeddings

Runs two training experiments back-to-back:
  Run A: Sinusoidal PE  (original PositionalEncoding from model.py)
  Run B: Learned PE     (torch.nn.Embedding — same shape, trained parameters)

Both runs use identical model architecture, dataset, seed, and optimizer.
Logs train_loss, val_loss, val_bleu per epoch to W&B for comparison.

"""

import math
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
NUM_EPOCHS   = 30
D_MODEL      = 256
N_LAYERS     = 3
NUM_HEADS    = 8
D_FF         = 512
DROPOUT      = 0.1
WARMUP_STEPS = 4000
MAX_LEN      = 256   # max sequence length for learned PE embedding table
BLEU_EVERY   = 2


# Reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Learned PE variant (subclass — no model.py edits)

class TransformerLearnedPE(Transformer):
    """
    Identical to Transformer but replaces sinusoidal PositionalEncoding
    with a learned nn.Embedding over positions.

    Only encode() and decode() are overridden — everything else
    (encoder stack, decoder stack, loss, optimizer, BLEU eval) is
    inherited unchanged.
    """

    def __init__(self, max_len: int = MAX_LEN, **kwargs):
        super().__init__(**kwargs)

        # Replace the sinusoidal PE with two learned position embeddings
        # (one for src, one for tgt — they share d_model but are independent)
        # Using two separate embeddings mirrors how BERT/GPT handle this.
        self.src_pos_embedding = nn.Embedding(max_len, self.d_model)
        self.tgt_pos_embedding = nn.Embedding(max_len, self.d_model)
        self.pos_dropout       = nn.Dropout(p=self.dropout)
        self.max_len           = max_len

        # Initialise with small normal weights (standard practice)
        nn.init.normal_(self.src_pos_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.tgt_pos_embedding.weight, mean=0, std=0.01)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Same as Transformer.encode() but uses learned src position embeddings.
        """
        batch, src_len = src.shape
        positions = torch.arange(src_len, device=src.device).unsqueeze(0)  # [1, src_len]

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)        # [batch, src_len, d_model]
        src_emb = src_emb + self.src_pos_embedding(positions)              # add learned PE
        src_emb = self.pos_dropout(src_emb)

        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as Transformer.decode() but uses learned tgt position embeddings.
        """
        batch, tgt_len = tgt.shape
        positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0)  # [1, tgt_len]

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)        # [batch, tgt_len, d_model]
        tgt_emb = tgt_emb + self.tgt_pos_embedding(positions)              # add learned PE
        tgt_emb = self.pos_dropout(tgt_emb)

        decoder_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.output_projection(decoder_output)


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
    use_learned:   bool,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    train_dataset: Multi30kDataset,
    val_dataset:   Multi30kDataset,
    device:        torch.device,
):
    """
    Args:
        use_learned : True  → learned PE (TransformerLearnedPE)
                      False → sinusoidal PE (original Transformer)
    """

    config = {
        "pe_type":     "learned_embedding" if use_learned else "sinusoidal",
        "d_model":     D_MODEL,
        "N":           N_LAYERS,
        "num_heads":   NUM_HEADS,
        "d_ff":        D_FF,
        "dropout":     DROPOUT,
        "warmup_steps": WARMUP_STEPS,
        "batch_size":  BATCH_SIZE,
        "num_epochs":  NUM_EPOCHS,
        "seed":        SEED,
        "max_len":     MAX_LEN if use_learned else "N/A",
    }

    wandb.init(
        project="da6401_assignment3",
        group="experiment_2_4_pe_ablation",
        name=run_name,
        config=config,
        reinit=True,
    )

    set_seed(SEED)

    #  Build model 
    if use_learned:
        model = TransformerLearnedPE(
            max_len=MAX_LEN,
            d_model=D_MODEL,
            N=N_LAYERS,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            dropout=DROPOUT,
        ).to(device)
    else:
        model = Transformer(
            d_model=D_MODEL,
            N=N_LAYERS,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            dropout=DROPOUT,
        ).to(device)

    tgt_vocab_size = train_dataset.tgt_vocab['vocab_size']
    tgt_pad_idx    = train_dataset.tgt_vocab['pad_idx']

    # Count trainable parameters — useful for report
    total_params    = sum(p.numel() for p in model.parameters())
    pe_params       = 0
    if use_learned:
        pe_params = (model.src_pos_embedding.weight.numel() +
                     model.tgt_pos_embedding.weight.numel())
    print(f"\n[{run_name}] Total params: {total_params:,}  |  PE params: {pe_params:,}")
    wandb.config.update({"total_params": total_params, "pe_params": pe_params})

    #  Loss / optimizer / scheduler 
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

        # BLEU every BLEU_EVERY epochs
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
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        }
        if val_bleu is not None:
            log_dict["val_bleu"] = val_bleu

        wandb.log(log_dict)

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | lr={current_lr:.2e}"
        )

    print(f"\n[{run_name}] Best val BLEU = {best_val_bleu:.2f}\n")
    wandb.finish()


# Entry point
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    # Run A: Sinusoidal PE 
    print("\n" + "="*60)
    print("RUN A: Sinusoidal Positional Encoding")
    print("="*60)
    run_experiment(
        run_name="sinusoidal_pe",
        use_learned=False,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    # Run B: Learned PE 
    print("\n" + "="*60)
    print("RUN B: Learned Positional Embeddings")
    print("="*60)
    run_experiment(
        run_name="learned_pe",
        use_learned=True,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    print("\nBoth runs complete.")
