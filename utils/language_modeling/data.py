import logging
import numpy as np
import random

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

TARGET_IDX = -100
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainDataset(Dataset):
    
    def __init__(self, data, maxlen=256, keep_first=False):
        super().__init__()
        self._data = data.values
        self._maxlen = maxlen
        self._keep_first = keep_first
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        if len(self._data[idx]) > self._maxlen:
            if self._keep_first:
                start = 0
            else:
                start = np.random.choice(len(self._data[idx]) - self._maxlen)
            seq = self._data[idx][start:start + self._maxlen]
        else:
            seq = self._data[idx]

        permuted = random.random() > 0.5
        if permuted:
            seqlen = len(seq) - 1
            seq = seq[0] + seq[seqlen // 2:] + seq[1:seqlen // 2]
        return torch.tensor(seq, dtype=torch.long), permuted


class Collator:

    def __init__(self, tokenizer, mask_prob=0.15, random_prob=0.2):
        self._vocab_size = tokenizer.vocab_size
        self._pad_token_id = tokenizer.pad_token_id
        self._mask_token_id = tokenizer.mask_token_id
        self._mask_prob = mask_prob
        self._random_prob = random_prob
        
    def __call__(self, batch):
        input_ids, permuted = zip(*batch)
        permuted = torch.tensor(permuted, dtype=torch.float)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)

        mask = torch.logical_and(
            torch.less(torch.rand(input_ids.shape), self._mask_prob),
            torch.not_equal(input_ids, self._pad_token_id)
        )
        truly_mask = torch.less(torch.rand(input_ids.shape), 1 - self._random_prob)
        random_mask = torch.less(torch.rand(input_ids.shape), 0.5)

        labels = torch.where(mask, input_ids, TARGET_IDX)

        # masking some of the tokens
        input_ids = torch.where(torch.logical_and(mask, truly_mask), self._mask_token_id, input_ids)

        # randomly changing other tokens
        input_ids = torch.where(
            torch.logical_and(mask, torch.logical_and(torch.logical_not(truly_mask), random_mask)),
            torch.randint_like(input_ids, low=5, high=self._vocab_size),
            input_ids
        )

        return input_ids, labels, permuted
    
    
def get_dataloaders(data, batch_size, collator, maxlen=256, test_size=0.1, seed=19):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    dataloaders = {
        'train': DataLoader(
            TrainDataset(train, maxlen), 
            batch_size=batch_size, 
            collate_fn=collator, 
            shuffle=True
        ),
        'eval': DataLoader(
            TrainDataset(test, maxlen, keep_first=True),
            batch_size=batch_size, 
            collate_fn=collator
        )
    }
    logger.info('\t Training epoch is {} steps long, {} samples.'.format(len(dataloaders['train']), train.shape[0]))
    logger.info('\t Validation is {} steps long, {} samples'.format(len(dataloaders['eval']), test.shape[0]))
    return dataloaders
