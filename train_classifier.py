import os
import json
import pickle
import logging
from argparse import ArgumentParser

import torch

import utils
import utils.classification as clf

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    return parser.parse_args()


def filter_targets(df):
    tmp = df['target'].value_counts()
    bad_targets = set(tmp[tmp == 1].index)
    mask = ~df['target'].isin(bad_targets)
    df = df.loc[mask].reset_index(drop=True).copy()
    return df


if __name__ == '__main__':
    args = parse_args()

    with open(args.params, 'r') as f:
        params = json.load(f)

    model_name = 'H{}_L{}_bs{}_ml{}_g{}'.format(
        params['model']['hidden_size'],
        params['model']['num_hidden_layers'],
        params['data']['batch_size'] * params['optimization']['accum_steps'],
        params['data']['maxlen'],
        params['model']['num_groups']
    )
    if 'suffix' in params:
        model_name += '_{}'.format(params['suffix'])

    with open(os.path.join(params['callbacks']['params'], f'{model_name}.json'), 'w') as f:
        json.dump(params, f, indent=2)

    with open(params['data'].pop('path'), 'rb') as f:
        train = pickle.load(f)

    train = filter_targets(train)

    tokenizer = utils.WordpieceTokenizer(**params['data'].pop('tokenizer'))
    label_encoder = utils.LabelEncoder.from_file(params['data'].pop('label_encoder'))

    # creating train and validation dataloaders with random split between records
    dataloaders = clf.get_dataloaders(
        data=train,
        tokenizer=tokenizer,
        **params['data']
    )

    # creating model
    model = clf.get_model(params['model'], tokenizer, label_encoder)

    optimizer = utils.get_optimizer(
        model=model,
        lr=params['optimization']['peak_lr'],
        weight_decay=params['optimization']['weight_decay'],
        head_factor=params['optimization']['head_factor']
    )

    # creating scheduler with linear warmup
    scheduler = utils.get_scheduler(
        optimizer=optimizer,
        dataloader=dataloaders['train'],
        num_epochs=params['optimization']['num_epochs'],
        accum_steps=params['optimization']['accum_steps'],
        warmup_params=params['optimization']['warmup']
    )

    # creating trainer
    log_dir = None
    if 'logs' in params['callbacks']:
        log_dir = os.path.join(params['callbacks']['logs'], model_name)
        logger.info('\t Logs are saved into {}'.format(log_dir))

    checkpoints = None
    if 'checkpoints' in params['callbacks']:
        checkpoints = os.path.join(params['callbacks']['checkpoints'], model_name)
        logger.info('\t Checkpoints are saved into {}'.format(checkpoints))

    assert torch.cuda.is_available(), 'CUDA should be available for this script.'
    device = torch.device('cuda')
    trainer = clf.Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logdir=log_dir,
        checkpoints=checkpoints,
        max_grad_norm=params['optimization']['max_grad_norm'],
        accum_steps=params['optimization']['accum_steps'],
        pad_token_id=tokenizer.pad_token_id
    )

    # training model
    trainer.train(dataloaders, n_epochs=params['optimization']['num_epochs'])
