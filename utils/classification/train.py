import os
import shutil

from apex import amp

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    
    def __init__(
            self, model, 
            optimizer, 
            scheduler,
            pad_token_id,
            device=None, 
            logdir=None, 
            opt_level='O1', 
            weights=None,
            max_grad_norm=0,
            checkpoints=None,
            accum_steps=1,
    ):
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._scheduler = scheduler
        self._model, self._optimizer = amp.initialize(model.to(self._device), optimizer, opt_level=opt_level)
        if weights is not None:
            weights = weights.to(self._device)
        self._criterion = torch.nn.CrossEntropyLoss(weight=weights)
        
        self._writer = None
        if logdir is not None:
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self._writer = SummaryWriter(log_dir=logdir)

        self._checkpoints = checkpoints
        if checkpoints is not None:
            if os.path.exists(checkpoints):
                shutil.rmtree(checkpoints)
            os.mkdir(checkpoints)

        self._n_batch = 1
        self._n_epoch = 0

        self._max_grad_norm = max_grad_norm
        self._accum_steps = accum_steps
        self._pad_token_id = pad_token_id

    def _train_step(self, dataloader):
        self._model.train()
        self._optimizer.zero_grad()

        accumulated_loss = 0
        for batch in dataloader:
            input_ids, targets = map(lambda x: x.to(self._device), batch)
            preds = self._model(input_ids, attention_mask=input_ids != self._pad_token_id)
            loss = self._criterion(preds, targets) / self._accum_steps
            accumulated_loss += loss.item()
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            if self._n_batch % self._accum_steps == 0:
                self._update_weights(accumulated_loss)
                accumulated_loss = 0
            self._n_batch += 1

    def _update_weights(self, loss):
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
            self._writer.add_scalar(
                'train/batch', loss, global_step=self._n_batch // self._accum_steps
            )
            if self._n_batch // self._accum_steps % 100 == 0:
                self._writer.add_scalar(
                    'optimization/lr', self._scheduler.get_last_lr()[0], global_step=self._n_batch // self._accum_steps
                )

    def _eval_step(self, dataloader):
        self._model.eval()
        
        total_loss = 0
        total_amount = 0
        for batch in dataloader:
            input_ids, targets = map(lambda x: x.to(self._device), batch)
            with torch.no_grad():
                preds = self._model(input_ids, attention_mask=input_ids != self._pad_token_id)
                loss = self._criterion(preds, targets)
            total_loss += loss.item() * len(batch)
            total_amount += len(batch)
            
        return total_loss / total_amount
    
    def train(self, dataloaders, n_epochs):
        for epoch in range(n_epochs):
            self._train_step(dataloaders['train'])

            if 'eval' in dataloaders:
                val_loss = self._eval_step(dataloaders['eval'])
                if self._writer is not None:
                    self._writer.add_scalar('eval', val_loss, global_step=self._n_epoch)

            if 'evaluator' in dataloaders:
                for metric, score in dataloaders['evaluator'](self).items():
                    self._writer.add_scalar(f'eval/{metric}', score, global_step=self._n_epoch)

            if self._checkpoints is not None:
                torch.save(
                    self.state_dict(),
                    os.path.join(self._checkpoints, 'epoch_{}.pt'.format(self._n_epoch + 1))
                )

            self._n_epoch += 1

    def state_dict(self):
        return self._model.state_dict()

    def predict(self, dataloader):
        preds = []
        for input_ids in dataloader:
            input_ids = input_ids.to(self._device)
            with torch.no_grad():
                pred = self._model(input_ids, attention_mask=input_ids != self._pad_token_id)
                _, indices = pred.max(dim=1)
            preds += indices.cpu().numpy().tolist()
        return preds