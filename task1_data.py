import json
import pickle
import tqdm
import logging
import pandas as pd
from argparse import ArgumentParser

import utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UNK_CATEGORY = -1


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.params, 'r') as f:
        params = json.load(f)

    train = pd.read_parquet(params['data_path']) \
        .drop_duplicates('item_name').reset_index(drop=True)

    # training tokenizer
    if not params['tokenizer'].pop('pretrained', False):
        tokenizer = utils.WordpieceTokenizer.from_corpus(
            corpus=train['item_name'].values,
            **params['tokenizer']
        )
    else:
        tokenizer = utils.WordpieceTokenizer(**params['tokenizer'], from_pretrained=True)

    # creating pretraining data
    tqdm.tqdm.pandas()
    train['ids'] = train['item_name'].progress_apply(
        lambda x: tokenizer.encode(x, add_cls_token=True)
    )

    with open(params['pretrain_save_path'], 'wb') as f:
        pickle.dump(train[['ids']], f)

    # creating finetuning data
    train = pd.read_parquet(params['data_path'])
    train = train[train.category_id != UNK_CATEGORY]
    train = train.groupby(['item_name', 'category_id'], sort=False).size().reset_index(name='count')

    enc = utils.LabelEncoder.from_corpus(train['category_id'])
    enc.save(params['label_encoder_save_path'])
    train['target'] = enc(train['category_id'])

    tqdm.tqdm.pandas()
    train['ids'] = train['item_name'].progress_apply(
        lambda x: tokenizer.encode(x, add_cls_token=True)
    )

    with open(params['finetune_save_path'], 'wb') as f:
        pickle.dump(train[['item_name', 'ids', 'target', 'count']], f)
