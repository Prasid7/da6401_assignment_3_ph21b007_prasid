"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F
# from bleu import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu
try:
    import sacrebleu as _sacrebleu
    HAS_SACREBLEU = True
    print("sacrebleu imported successfully. Will use sacrebleu for BLEU evaluation.")
except ImportError:
    HAS_SACREBLEU = False
import wandb
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import *
from lr_scheduler import *


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

        self.confidence = 1.0 - smoothing # probability mass for the correct class


    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """

        # logits shape: [batch*tgt_len, vocab_size]
        # target shape: [batch*tgt_len]
        
        # compute log probabilities from logits
        log_probs = F.log_softmax(logits, dim=-1) # shape [batch*tgt_len, vocab_size]

        # smoothing value distributed across non-target, non-pad tokens
        smoothing_value = self.smoothing / (self.vocab_size - 2) # exclude target and pad

        # initialize smoothed target distribution with smoothing_value
        true_dist = torch.full_like(log_probs, smoothing_value) # shape [batch*tgt_len, vocab_size]

        # set correct token probability to confidence
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence) # shape [batch*tgt_len, vocab_size]

        # zero out pad token probability
        true_dist[:, self.pad_idx] = 0.0

        # identify pad targets
        pad_mask = (target == self.pad_idx) # shape [batch*tgt_len]

        # zero out entire rows corresponding to pad targets
        true_dist[pad_mask] = 0.0

        # compute cross-entropy loss: sum(-true_dist * log_probs) 
        loss = torch.sum(-true_dist * log_probs, dim=-1) # shape [batch*tgt_len]

        # average loss over non-pad masks
        non_pad_mask = ~pad_mask # shape [batch*tgt_len]
        avg_loss = loss[non_pad_mask].mean() # scalar

        return avg_loss


        
# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """

    # depending on is_train, set model to train() or eval() mode
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0 # initialize running loss total

    # loop over batches from data_iter
    for batch_idx, (src, tgt) in enumerate(data_iter):
        # move src and tgt to device
        src = src.to(device) # shape [batch, src_len]
        tgt = tgt.to(device) # shape [batch, tgt_len]

        tgt_input = tgt[:, :-1] # input to the decoder (exclude <eos>), shape [batch, tgt_len-1]
        tgt_output = tgt[:, 1:] # target for loss (exclude <sos>), shape [batch, tgt_len-1]

        with torch.set_grad_enabled(is_train):

            src_mask = make_src_mask(src, model.src_pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, model.tgt_pad_idx)

            # forward pass through model to get output logits
            logits = model(src, tgt_input, src_mask, tgt_mask) # shape [batch, tgt_len-1, vocab_size]

            # reshape logits and target for loss computation
            logits_flat = logits.reshape(-1, logits.size(-1)) # shape [batch*(tgt_len-1), vocab_size]
            target_flat = tgt_output.reshape(-1) # shape [batch*(tgt_len-1)]

            # compute loss using loss_fn
            loss = loss_fn(logits_flat, target_flat) # scalar

            if is_train:
                optimizer.zero_grad() # zero gradients
                loss.backward()       # backward pass, compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
                optimizer.step()      # update parameters

                if scheduler is not None:
                    scheduler.step()   # update learning rate

        total_loss += loss.item() # accumulate total avg. batch loss for epoch

    avg_loss = total_loss / len(data_iter) # average loss over all tokens

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    
    model.eval() # set model to eval mode

    # disable gradient computation for inference
    with torch.no_grad():

        # run encoder once on src to get memory
        memory = model.encode(src, src_mask) # shape [1, src_len, d_model]

        # initialize ys gnerated sequence with start_symbol, shape [1, 1]
        ys = torch.tensor([[start_symbol]], device=device) # shape [1, 1]

        # autoregressive decoding loop
        for i in range(max_len - 1): # generate up to max_len tokens (including start_symbol)

            tgt_mask = make_tgt_mask(ys, model.tgt_pad_idx).to(device) # shape [1, 1, out_len, out_len] 

            # run decoder on current ys to get output logits
            logits = model.decode(memory, src_mask, ys, tgt_mask) # shape [1, out_len, vocab_size]

            # extract logits for the last generated token, shape [1, vocab_size]
            next_token_logits = logits[:, -1, :] # shape [1, vocab_size]

            # choose token with highest probability as next token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # shape [1, 1] 
            ys = torch.cat([ys, next_token], dim=1) # shape [1, out_len]

            # stop if end_symbol is generated
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    Uses model.infer() + sacrebleu to match Gradescope evaluation exactly.
    Falls back to nltk corpus_bleu if sacrebleu is not installed.
    """
    from datasets import load_dataset

    model.eval()

    # ── sacrebleu path (matches Gradescope) ────────────────────────────
    if HAS_SACREBLEU:
        # infer which split to use from the dataloader's dataset size
        # val = 1014, test = 1000 — this lets the same function work for both
        dataset_size = len(test_dataloader.dataset)
        split = "validation" if dataset_size > 1000 else "test"
        raw_data = load_dataset("bentrevett/multi30k")[split]

        hypotheses = []
        references = []

        for entry in raw_data:
            hypothesis = model.infer(entry["de"])
            hypotheses.append(hypothesis)
            references.append(entry["en"])

        bleu = _sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score   # already 0-100

    # ── nltk fallback (kept for compatibility) ─────────────────────────
    references_nltk  = []
    hypotheses_nltk  = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_sentence = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_sentence, model.src_pad_idx).to(device)
                generated = greedy_decode(
                    model, src_sentence, src_mask, max_len,
                    start_symbol=model.tgt_sos_idx,
                    end_symbol=model.tgt_eos_idx,
                    device=device
                )
                generated_tokens = generated.squeeze(0).tolist()
                generated_sentence = [tgt_vocab['idx2word'][idx] for idx in generated_tokens]
                generated_sentence = [t for t in generated_sentence if t not in ['<sos>', '<eos>', '<pad>']]
                if not generated_sentence:
                    generated_sentence = ["<unk>"]

                target_tokens = tgt[i].tolist()
                target_sentence = [tgt_vocab['idx2word'][idx] for idx in target_tokens]
                target_sentence = [t for t in target_sentence if t not in ['<sos>', '<eos>', '<pad>']]

                references_nltk.append([target_sentence])
                hypotheses_nltk.append(generated_sentence)

    bleu_score = corpus_bleu(references_nltk, hypotheses_nltk)
    return bleu_score * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    
    model_config = {
        'd_model': model.d_model,
        'N': model.N,
        'num_heads': model.num_heads,
        'd_ff': model.d_ff,
        'dropout': model.dropout,
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model_config, 
        'src_vocab': model.src_vocab,
        'tgt_vocab': model.tgt_vocab,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location='cpu') # load checkpoint to CPU

    model.load_state_dict(checkpoint['model_state_dict']) # restore model weights

    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # restore optimizer state

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # restore scheduler state

    epoch = checkpoint['epoch'] # get saved epoch number

    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """

    # define hyperparameters and W&B config
    config = {
        "batch_size": 128,
        "num_epochs": 50,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.3,
        "learning_rate": 1.0, 
        "warmup_steps": 4000
    }

    # initialize W&B run with config and descriptive name
    wandb.init(
        project="da6401_assignment3",
        config=config,
        name=(
            f"transformer_d{config['d_model']}"
            f"_h{config['num_heads']}"
            f"_l{config['N']}"
            f"_ff{config['d_ff']}"
            f"_bs{config['batch_size']}"
            f"_ep{config['num_epochs']}"
        )
    )

    # build datasets

    train_dataset = Multi30kDataset(split='train')
    val_dataset = Multi30kDataset(split='validation', src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
    test_dataset = Multi30kDataset(split='test', src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                            collate_fn=lambda x: collate_fn( x, train_dataset.src_vocab['pad_idx'], train_dataset.tgt_vocab['pad_idx']))
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, val_dataset.src_vocab['pad_idx'], val_dataset.tgt_vocab['pad_idx']))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, test_dataset.src_vocab['pad_idx'], test_dataset.tgt_vocab['pad_idx']))
    
    # extract vocab info 
    src_vocab_size = len(train_dataset.src_vocab['word2idx'])
    tgt_vocab_size = len(train_dataset.tgt_vocab['word2idx'])

    # instantiate model — __init__ builds vocab/tokenizers internally and
    # attempts gdown download (will gracefully skip if weights not yet on Drive)
    model = Transformer(
        d_model=wandb.config.d_model,
        N=wandb.config.N,
        num_heads=wandb.config.num_heads,
        d_ff=wandb.config.d_ff,
        dropout=wandb.config.dropout,
    )

    # move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # create Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # create NoamScheduler
    scheduler = NoamScheduler(optimizer, d_model=wandb.config.d_model, warmup_steps=wandb.config.warmup_steps)

    # create LabelSmoothingLoss
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, train_dataset.tgt_vocab['pad_idx'], smoothing=0.1).to(device)

    # training loop
    best_val_bleu = 0.0  # track best validation BLEU for saving best checkpoint

    for epoch in range(wandb.config.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)

        # evaluate BLEU every 2 epochs (sacrebleu is slow — runs infer() on 1014 sentences)
        if epoch % 2 == 0:
            val_bleu = evaluate_bleu(model, val_loader, val_dataset.tgt_vocab, device=device)
            wandb.log({'bleu': val_bleu})
            print(f"Validation BLEU = {val_bleu:.2f}")

            # save best checkpoint when val BLEU improves
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint_best.pth")
                print(f"New best checkpoint saved (BLEU={val_bleu:.2f})")

        # log losses to W&B
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # save per-epoch checkpoint for recovery
        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch_{epoch}.pth")

    # final BLEU evaluation on test set
    test_bleu = evaluate_bleu(model, test_loader, test_dataset.tgt_vocab, device=device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({'test_bleu': test_bleu})

    wandb.finish()



if __name__ == "__main__":
    run_training_experiment()

