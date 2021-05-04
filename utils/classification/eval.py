import logging

from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class InferenceDataset(Dataset):

    def __init__(self, data, maxlen):
        super().__init__()
        self._ids = data['ids'].values
        self._maxlen = maxlen
        
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self._ids[idx][:self._maxlen])
    

class Collator:

    def __init__(self, tokenizer):
        self._pad_token_id = tokenizer.pad_token_id

    def __call__(self, input_ids):
        return pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)


class Evaluator:
    
    def __init__(self, data, tokenizer, batch_size, maxlen, scorer):
        self._dataset = InferenceDataset(data, maxlen)
        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=batch_size, 
            collate_fn=Collator(tokenizer)
        )
        self._targets = data['target'].values
        self._scorer = scorer

    def __call__(self, trainer, verbose=False):
        preds = trainer.predict(self._dataloader)
        return {
            'weighted_f1': f1_score(self._targets, preds, average='weighted'),
            'weighted_f1_true': self._scorer(self._targets, preds)
        }
