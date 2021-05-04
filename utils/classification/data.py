import logging

from sklearn.model_selection import train_test_split

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    Dataset,
    DataLoader
)

from utils.metrics import F1WeightedScorer
from utils.classification.eval import Evaluator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FinetuneDataset(Dataset):

    def __init__(self, data, maxlen):
        super().__init__()
        self._ids = data['ids'].values
        self._targets = data['target'].values
        self._maxlen = maxlen
        
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self._ids[idx][:self._maxlen]), \
               self._targets[idx]
    

class Collator:

    def __init__(self, tokenizer):
        self._pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids, targets = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)
        return input_ids, \
            torch.tensor(targets, dtype=torch.long)


def get_dataloaders(data, tokenizer, batch_size, maxlen, test_size=0.1, seed=19):
    if test_size > 0.:
        train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data['target'])
        test = test.copy()
        test['length'] = test['ids'].map(len)
        test.sort_values('length', inplace=True)
    else:
        train = data
        test = None

    class_weights = data['target'].value_counts().sort_index().values
    class_weights = class_weights / class_weights.sum()

    collator = Collator(tokenizer)
    dataloaders = {
        'train': DataLoader(
            dataset=FinetuneDataset(train, maxlen),
            batch_size=batch_size, 
            collate_fn=collator,
            shuffle=True
        )
    }
    logger.info('\t Training epoch is {} steps long, {} samples.'.format(len(dataloaders['train']), train.shape[0]))
    if test_size > 0.:
        dataloaders.update({
            'eval': DataLoader(
                dataset=FinetuneDataset(test, maxlen),
                batch_size=batch_size,
                collate_fn=collator
            ),
            'evaluator': Evaluator(test, tokenizer, batch_size, maxlen, F1WeightedScorer(class_weights))
        })
        logger.info('\t Validation is {} steps long, {} samples'.format(len(dataloaders['eval']), test.shape[0]))
    return dataloaders