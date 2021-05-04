import logging

import torch
from torch import nn

from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertLayer
)
from transformers.activations import ACT2FN

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CLSModel(nn.Module):

    def __init__(self, config, embedding_dim, num_groups, num_classes, msd=None):
        super().__init__()
        self.backbone = Backbone(config, embedding_dim, num_groups)
        self.sop_head = CLSHead(config, num_classes, msd)

    def forward(self, x, attention_mask):
        hidden_states = self.backbone(x, attention_mask)
        return self.sop_head(hidden_states)

    def load_weights(self, path):
        found = []
        with open(path, 'rb') as f:
            weights = torch.load(f)
        for name, param in weights['model'].items():
            if name in self.state_dict() and param.shape == self.state_dict()[name].shape:
                self.state_dict()[name].copy_(param)
                logger.info(f'\t Preloading {name}')
                found.append(name)

        logger.info('\n\t Didnt find layers:')
        for name in self.state_dict():
            if name not in weights['model']:
                logger.info(f'\t {name}')

        return found


class CLSHead(nn.Module):
    CLS_POSITION = 0

    def __init__(self, config, num_classes, msd=None):
        super().__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.msd = None
        if msd is not None:
            self.msd_dropout = nn.Dropout(p=msd['prob'])
            self.msd_num_dropouts = msd['num']

    def forward(self, hidden_states):
        x = self.pooler(hidden_states[:, self.CLS_POSITION])
        x = self.pooler_activation(x)
        x = self.dropout(x)
        if self.msd is not None:
            out = self.classifier(self.msd_dropout(x))
            for _ in range(self.msd_num_dropouts - 1):
                out += self.classifier(self.msd_dropout(x))
            out /= self._n_dropouts
        else:
            out = self.classifier(x)
        return out


class SOPHead(nn.Module):
    CLS_POSITION = 0
    CRITERION = nn.BCEWithLogitsLoss()

    def __init__(self, config):
        super().__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, permuted):
        x = self.pooler(hidden_states[:, self.CLS_POSITION])
        x = self.pooler_activation(x)
        x = self.dropout(x)
        x = self.classifier(x)
        preds = torch.squeeze(x, dim=-1)
        return self.CRITERION(preds, permuted)


class AlbertModel(nn.Module):

    def __init__(self, config, embedding_dim, num_groups):
        super().__init__()
        self.backbone = Backbone(config, embedding_dim, num_groups)
        self.mlm_head = MLMHead(config)
        self.sop_head = SOPHead(config)

    def forward(self, x, attention_mask, labels, permuted):
        hidden_states = self.backbone(x, attention_mask)
        mlm_loss = self.mlm_head(hidden_states, labels)
        sop_loss = self.sop_head(hidden_states, permuted)
        return 0.5 * mlm_loss + 0.5 * sop_loss, {'MLM': mlm_loss, 'SOP': sop_loss}


class MaskedLanguageModel(nn.Module):

    def __init__(self, config, embedding_dim, num_groups):
        super().__init__()
        self.backbone = Backbone(config, embedding_dim, num_groups)
        self.head = MLMHead(config)

    def forward(self, x, attention_mask, labels):
        hidden_states = self.backbone(x, attention_mask)
        return self.head(hidden_states, labels)


class MLMHead(nn.Module):
    CRITERION = nn.CrossEntropyLoss()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states, labels):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        preds = self.decoder(hidden_states)
        return self.CRITERION(preds.view(-1, self.config.vocab_size), labels.view(-1))


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
