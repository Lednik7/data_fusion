import json
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertLayer
)

from transformers import BertConfig

from transformers import BertTokenizerFast

INTERMEDIATE_MULTIPLIER = 4
SINGLE_HEAD_SIZE = 64


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


class Backbone(BertPreTrainedModel):

    def __init__(self, config, embedding_dim, num_groups):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config, embedding_dim)
        msg = 'Amount of encoder blocks should be divisible by number of groups.'
        assert config.num_hidden_layers % num_groups == 0, msg
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(num_groups)])
        self.group_size = config.num_hidden_layers // num_groups
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        input_shape = input_ids.size()
        batch_size, _ = input_shape

        hidden_states = self.embeddings(input_ids)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.shape, device)
        for encoder_block in self.encoder:
            for _ in range(self.group_size):
                hidden_states = encoder_block(hidden_states, extended_attention_mask)[0]

        return hidden_states


class BertEmbeddings(nn.Module):

    def __init__(self, config, embedding_dim):
        super().__init__()
        if embedding_dim != config.hidden_size:
            self.projection = nn.Linear(embedding_dim, config.hidden_size)
        else:
            self.projection = nn.Identity()
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, input_ids):
        _, seq_length = input_ids.size()

        embeddings = self.projection(self.word_embeddings(input_ids)) \
             + self.position_embeddings(self.position_ids[:, :seq_length])
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class CLSModel(nn.Module):

    def __init__(self, config, embedding_dim, num_groups, num_classes):
        super().__init__()
        self.backbone = Backbone(config, embedding_dim, num_groups)
        self.sop_head = CLSHead(config, num_classes)

    def forward(self, x, attention_mask):
        hidden_states = self.backbone(x, attention_mask)
        return self.sop_head(hidden_states)


class CLSHead(nn.Module):
    CLS_POSITION = 0

    def __init__(self, config, num_classes):
        super().__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.pooler(hidden_states[:, self.CLS_POSITION])
        x = self.pooler_activation(x)
        x = self.dropout(x)
        return self.classifier(x)


def get_model(params, tokenizer, vocab_size):
    embedding_dim = params.pop('embedding_dim', params['hidden_size'])
    num_groups = params.pop('num_groups', params['num_hidden_layers'])
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_attention_heads=params['hidden_size'] // SINGLE_HEAD_SIZE,
        intermediate_size=params['hidden_size'] * INTERMEDIATE_MULTIPLIER,
        output_hidden_states=True,
        **params
    )
    return CLSModel(cfg, embedding_dim, num_groups, vocab_size)


class WordpieceTokenizer:

    def __init__(self, vocab_path, strip_accents, clean_text, lowercase):
        common_params = {'strip_accents': strip_accents, 'clean_text': clean_text, 'lowercase': lowercase}
        self._tokenizer = BertTokenizerFast(
            vocab_file=vocab_path, **common_params
        )

    def encode(self, text, add_cls_token=True):
        ids = []
        if add_cls_token:
            ids.append(self.cls_token_id)
        ids.extend(self._tokenizer.encode(text, add_special_tokens=False))
        return ids

    @property
    def cls_token_id(self):
        return self._tokenizer.cls_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def sep_token_id(self):
        return self._tokenizer.sep_token_id

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size


if __name__ == '__main__':
    with open('inference.json', 'r') as f:
        params = json.load(f)

    tokenizer = WordpieceTokenizer(**params['data']['tokenizer'])

    test = pd.read_parquet(params['data']['path'])
    test['ids'] = test['item_name'].map(
        lambda x: tokenizer.encode(x, add_cls_token=True)
    )

    with open(params['data']['label_encoder'], 'rb') as f:
        vocab2id = pickle.load(f)

    # creating transformer encoder model
    model = get_model(params['model'], tokenizer, len(vocab2id))

    device = torch.device('cuda')
    model = model.to(device)

    checkpoint = torch.load(params['data']['checkpoint'])
    model.load_state_dict(checkpoint)

    ds = InferenceDataset(test, params['data']['maxlen'])
    dataloader = DataLoader(ds, batch_size=params['data']['batch_size'], collate_fn=Collator(tokenizer))

    model.eval()
    preds = []
    for input_ids in dataloader:
        input_ids = input_ids.to(device)
        with torch.no_grad():
            pred = model(input_ids, attention_mask=input_ids != tokenizer.pad_token_id)
            _, indices = pred.max(dim=1)
        preds += indices.cpu().numpy().tolist()
    preds = np.array(preds)

    id2category = {idx: category for category, idx in vocab2id.items()}

    res = pd.DataFrame(preds, columns=['pred'])
    res['pred'] = res['pred'].map(id2category)
    res['id'] = test['id']
    res[['id', 'pred']].to_csv(params['data']['save'], index=None)
