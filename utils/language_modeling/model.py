from transformers import (
    BertForMaskedLM,
    BertConfig
)
from utils import AlbertModel

INTERMEDIATE_MULTIPLIER = 4
SINGLE_HEAD_SIZE = 64


def get_model(params, tokenizer):
    if 'huggingface' in params:
        return BertForMaskedLM.from_pretrained(params['huggingface'])

    embedding_dim = params.pop('embedding_dim', params['hidden_size'])
    num_groups = params.pop('num_groups', params['num_hidden_layers'])
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_attention_heads=params['hidden_size'] // SINGLE_HEAD_SIZE,
        intermediate_size=params['hidden_size'] * INTERMEDIATE_MULTIPLIER,
        output_hidden_states=False,
        **params
    )
    return AlbertModel(cfg, embedding_dim, num_groups)
