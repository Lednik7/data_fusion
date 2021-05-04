import os
import shutil
from collections import defaultdict

from apex import amp

import torch
from torch.utils.tensorboard import SummaryWriter

TARGET_IDX = -100


class Trainer:

    def __init__(
            self, 
            model, 
            optimizer, 
            scheduler,
            pad_token_id,
            device=None, 
            logdir=None, 
            checkpoints=None,
            max_grad_norm=0.,
            accum_steps=1,
            opt_level='O1'
    ):
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._scheduler = scheduler
        self._opt_level = opt_level
        self._model, self._optimizer = amp.initialize(model.to(self._device), optimizer, opt_level=opt_level)

        self._writer = None
        if logdir is not None:
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self._writer = SummaryWriter(log_dir=logdir)
        self._n_batch = 1
        self._n_epoch = 0
        
        self._checkpoints = checkpoints
        if checkpoints is not None:
            if os.path.exists(checkpoints):
                shutil.rmtree(checkpoints)
            os.mkdir(checkpoints)

        self._max_grad_norm = max_grad_norm
        self._accum_steps = accum_steps
        self._pad_token_id = pad_token_id

    def train(self, dataloaders, n_epochs):
        for epoch in range(n_epochs):
            self._train_step(dataloaders['train'])
            val_losses = self._eval_step(dataloaders['eval'])

            if self._writer is not None:
                for name, loss in val_losses.items():
                    self._writer.add_scalar('eval/{}'.format(name), loss, global_step=self._n_epoch)
            if self._checkpoints is not None:
                torch.save(
                    self.state_dict(),
                    os.path.join(self._checkpoints, 'epoch_{}.pt'.format(self._n_epoch + 1))
                )
            self._n_epoch += 1

    def _train_step(self, dataloader):
        self._model.train()
        self._optimizer.zero_grad()
        accum_losses = defaultdict(float)
        for batch in dataloader:
            input_ids, labels, permuted = map(lambda x: x.to(self._device), batch)
            attention_mask = input_ids != self._pad_token_id
            total_loss, losses = self._model(input_ids, attention_mask, labels, permuted)
            for name, loss in losses.items():
                accum_losses[name] += loss.item() / self._accum_steps
            with amp.scale_loss(total_loss / self._accum_steps, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            if self._n_batch % self._accum_steps == 0:
                self._update_weights(accum_losses)
                for name in accum_losses:
                    accum_losses[name] = 0.
            self._n_batch += 1

    def _update_weights(self, losses):
        if self._max_grad_norm > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        else:
            grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(self._device) for p in self._model.parameters()]),
                2.0
            )
        if self._writer is not None:
            self._writer.add_scalar(
                'optimization/grad_norm', grad_norm.item(), global_step=self._n_batch // self._accum_steps
            )

        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

        if self._writer is not None:
            for name, loss in losses.items():
                self._writer.add_scalar(
                    'train/{}'.format(name), loss, global_step=self._n_batch // self._accum_steps
                )
            if self._n_batch // self._accum_steps % 100 == 0:
                self._writer.add_scalar(
                    'optimization/lr', self._scheduler.get_last_lr()[0], global_step=self._n_batch // self._accum_steps
                )

    def _eval_step(self, dataloader):
        self._model.eval()

        total_losses = defaultdict(float)
        total_amounts = defaultdict(int)
        for batch in dataloader:
            input_ids, labels, permuted = map(lambda x: x.to(self._device), batch)
            attention_mask = input_ids != self._pad_token_id
            with torch.no_grad():
                total_loss, losses = self._model(input_ids, attention_mask, labels, permuted)
            amount = (labels != -TARGET_IDX).sum().item()
            total_amounts['MLM'] += amount
            total_amounts['SOP'] += len(batch)
            total_losses['MLM'] += losses['MLM'].item() * amount
            total_losses['SOP'] += losses['SOP'].item() * len(batch)

        return {name: total_losses[name] / total_amounts[name] for name in total_losses}

    def state_dict(self):
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'amp': amp.state_dict(),
            'scheduler': self._scheduler.state_dict()
        }