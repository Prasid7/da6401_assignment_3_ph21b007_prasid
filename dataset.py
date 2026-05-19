import torch
from datasets import load_dataset
import spacy
from torch.utils.data import Dataset


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None,
    tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split

        # Load dataset from Hugging Face https://huggingface.co/datasets/bentrevett/multi30k
        self.dataset = load_dataset("bentrevett/multi30k")[self.split]

        # load spacy tokenizers for German and English
        self.de_tokenizer = spacy.load("de_core_news_sm")
        self.en_tokenizer = spacy.load("en_core_web_sm")
 
        # special tokens
        self.unk_token = "<unk>" # unknown token for out-of-vocabulary words
        self.pad_token = "<pad>" # padding token
        self.sos_token = "<sos>" # start of sentence
        self.eos_token = "<eos>" # end of sentence

        self.special_tokens = [self.unk_token, self.pad_token, self.sos_token, self.eos_token] 

        # Use provided vocabularies or build from training data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab() # build vocab from training data if not provided
        
        self.process_data() # process the data to convert sentences to token indices

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """

        train_data = load_dataset("bentrevett/multi30k")['train'] # we only build vocab from training data

        # Initialize vocab with special tokens
        src_word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        # src_idx2word = {idx: token for idx, token in enumerate(self.special_tokens)}

        tgt_word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        # tgt_idx2word = {idx: token for idx, token in enumerate(self.special_tokens)}

        # Build vocab from training data
        for entry in train_data:
            src_sentence = entry['de']
            tgt_sentence = entry['en']

            # Tokenize sentences using spacy
            src_tokens = [token.text for token in self.de_tokenizer(src_sentence)]
            tgt_tokens = [token.text for token in self.en_tokenizer(tgt_sentence)]

            # Add tokens to vocab if not already present
            for token in src_tokens:
                if token not in src_word2idx:
                    src_word2idx[token] = len(src_word2idx) # assign next available index
            
            for token in tgt_tokens:
                if token not in tgt_word2idx:
                    tgt_word2idx[token] = len(tgt_word2idx) # assign next available index
        
        # build reverse mapping as well
        src_idx2word = {idx: token for token, idx in src_word2idx.items()}
        tgt_idx2word = {idx: token for token, idx in tgt_word2idx.items()}

        # store vocabularies
        self.src_vocab = {
            'word2idx': src_word2idx,
            'idx2word': src_idx2word, 
            'vocab_size': len(src_word2idx),
            'unk_idx': src_word2idx[self.unk_token],
            'pad_idx': src_word2idx[self.pad_token],
            'sos_idx': src_word2idx[self.sos_token],
            'eos_idx': src_word2idx[self.eos_token]
        }

        self.tgt_vocab = {
            'word2idx': tgt_word2idx,
            'idx2word': tgt_idx2word,
            'unk_idx': tgt_word2idx[self.unk_token],
            'pad_idx': tgt_word2idx[self.pad_token],
            'sos_idx': tgt_word2idx[self.sos_token],
            'eos_idx': tgt_word2idx[self.eos_token],
            'vocab_size': len(tgt_word2idx)
        }


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """

        self.data = [] # list of (src_indices, tgt_indices) pairs

        src_word2idx = self.src_vocab['word2idx']
        tgt_word2idx = self.tgt_vocab['word2idx']

        src_unk_idx = self.src_vocab['unk_idx']
        tgt_unk_idx = self.tgt_vocab['unk_idx']
        
        for entry in self.dataset:
            src_sentence = entry['de']
            tgt_sentence = entry['en']

            # Tokenize sentences using spacy
            src_tokens = [token.text for token in self.de_tokenizer(src_sentence)]
            tgt_tokens = [token.text for token in self.en_tokenizer(tgt_sentence)]

            # add sos and eos tokens to sentences
            src_tokens = [self.sos_token] + src_tokens + [self.eos_token]
            tgt_tokens = [self.sos_token] + tgt_tokens + [self.eos_token]

            # convert token to ids
            src_indices = [src_word2idx.get(token, src_unk_idx) for token in src_tokens]
            tgt_indices = [tgt_word2idx.get(token, tgt_unk_idx) for token in tgt_tokens]

            # store the processed data
            self.data.append((src_indices, tgt_indices))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    """
    Collate function to be used with DataLoader for batching variable-length sequences.
    It pads the source and target sequences in the batch to the maximum length in the batch.
    """

    src_batch, tgt_batch = zip(*batch) # unzip the batch into src and tgt lists

    # find max length in the batch
    max_src_len = max(len(src) for src in src_batch)
    max_tgt_len = max(len(tgt) for tgt in tgt_batch)

    # pad the sequences
    padded_src_batch = [src + [src_pad_idx] * (max_src_len - len(src)) for src in src_batch]
    padded_tgt_batch = [tgt + [tgt_pad_idx] * (max_tgt_len - len(tgt)) for tgt in tgt_batch]

    # convert to tensors
    src_tensor = torch.tensor(padded_src_batch, dtype=torch.long)
    tgt_tensor = torch.tensor(padded_tgt_batch, dtype=torch.long)

    return src_tensor, tgt_tensor