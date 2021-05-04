import logging

import torch
from torch import nn

from transformers import (
    BertForSequenceClassification,
    BertConfig,
)
from utils.bert import CLSModel

INTERMEDIATE_MULTIPLIER = 4
SINGLE_HEAD_SIZE = 64

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiDropoutHead(nn.Module):

    def __init__(self, prob, n):
        super().__init__()
        self._dropout = nn.Dropout(p=prob)
        self._n_dropouts = n

    def forward(self, x, head):
        out = head(self._dropout(x))
        for _ in range(self._n_dropouts - 1):
            out += head(self._dropout(x))
        out /= self._n_dropouts
        return out


class Model(nn.Module):
    CLS_POSITION = 0

    def __init__(self, cfg=None, mdo_prob=0., mdo_num=1, num_classes=1, path=None):
        # unnecessary head is present
        super().__init__()
        if path is not None:
            self.backbone = BertForSequenceClassification.from_pretrained(path)
            self.backbone.config.output_hidden_states = True
        else:
            assert cfg is not None, 'Config should be provided if no pretrained path was specified.'
            self.backbone = BertForSequenceClassification(cfg)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

        weights_init = torch.zeros(self.backbone.config.num_hidden_layers).float()
        self.cls_weights = torch.nn.Parameter(weights_init, requires_grad=True)

        self.mdo = None
        if mdo_prob > 0.:
            self.mdo = MultiDropoutHead(mdo_prob, mdo_num)

    def forward(self, x, attention_mask):
        _, hidden_states = self.backbone(x, attention_mask)
        hidden_states = torch.stack([states[:, self.CLS_POSITION] for states in hidden_states[1:]])
        x = torch.einsum('ijk,i->jk', hidden_states, torch.softmax(self.cls_weights, dim=-1))

        if self.mdo is not None:
            return self.mdo(x, self.head)

        return self.head(x)

    def load_weights(self, path):
        found = []
        with open(path, 'rb') as f:
            weights = torch.load(f)
        for name, param in weights['model'].items():
            if name in self.backbone.state_dict() and 'cls' not in name:
                if param.shape == self.backbone.state_dict()[name].shape:
                    self.backbone.state_dict()[name].copy_(param)
                    logger.info(f'\t Preloading {name}')
                    found.append(name)

        logger.info('\n\t Didnt find layers:')
        for name in self.backbone.state_dict():
            if name not in weights['model']:
                logger.info(f'\t {name}')

        return found


def get_model(params, tokenizer, label_encoder):
    pretrained = params.pop('pretrained', None)

    embedding_dim = params.pop('embedding_dim', params['hidden_size'])
    num_groups = params.pop('num_groups', params['num_hidden_layers'])
    msd = params.pop('multisample_dropout', None)
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_attention_heads=params['hidden_size'] // SINGLE_HEAD_SIZE,
        intermediate_size=params['hidden_size'] * INTERMEDIATE_MULTIPLIER,
        output_hidden_states=True,
        **params
    )
    model = CLSModel(cfg, embedding_dim, num_groups, label_encoder.vocab_size, msd)
    if pretrained is not None:
        found = model.load_weights(pretrained)
        logger.info('\t Preloaded {} pretrained layers from {}.'.format(len(found), pretrained))
    else:
        logger.info('\t No pretrained weights provided.')

    return model
