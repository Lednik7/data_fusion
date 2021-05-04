import logging

import torch

import transformers

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NO_DECAY = ['bias', 'LayerNorm.weight']


def is_backbone(name):
    return 'backbone' in name# or 'pooler' in name


def needs_decay(name):
    return not any(word in name for word in NO_DECAY)


def get_optimizer(model, lr, weight_decay, head_factor):
    grouped_parameters = [
        {
            'params': [
                param for name, param in model.named_parameters() if is_backbone(name) and needs_decay(name)
            ],
            'lr': lr,
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param for name, param in model.named_parameters() if is_backbone(name) and not needs_decay(name)
            ],
            'lr': lr,
            'weight_decay': 0.,
        },
        {
            'params': [param for name, param in model.named_parameters() if not is_backbone(name)],
            'lr': lr * head_factor,
            'weight_decay': weight_decay,
        }
    ]

    logger.info(f'\t Head parameters with factor {head_factor}:')
    for name, _ in model.named_parameters():
        if not is_backbone(name):
            logger.info(f'\t \t {name}')

    return torch.optim.AdamW(grouped_parameters, lr=lr)


def get_scheduler(optimizer, dataloader, num_epochs, accum_steps, warmup_params):
    epoch_size = len(dataloader)
    num_training_steps = int(epoch_size * num_epochs / accum_steps)
    num_warmup_steps = warmup_params.get(
        'num_steps', int(num_training_steps * warmup_params['percentage'])
    )
    msg = '\t Linear warmup schedule with {} warmup steps out of {} total steps.'
    logger.info(msg.format(num_warmup_steps, num_training_steps))
    return transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
