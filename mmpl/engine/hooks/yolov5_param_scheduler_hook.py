import math
from typing import Optional
import numpy as np
from lightning import Callback
from mmengine.runner import Runner
from mmyolo.registry import HOOKS

from typing import Dict, Optional, Union

from mmengine.optim import _ParamScheduler
from mmengine.registry import HOOKS
from mmengine.utils import is_list_of

DATA_BATCH = Optional[Union[dict, tuple, list]]


def linear_fn(lr_factor: float, max_epochs: int):
    """Generate linear function."""
    return lambda x: (1 - x / max_epochs) * (1.0 - lr_factor) + lr_factor


def cosine_fn(lr_factor: float, max_epochs: int):
    """Generate cosine function."""
    return lambda x: (
        (1 - math.cos(x * math.pi / max_epochs)) / 2) * (lr_factor - 1) + 1


class ParamSchedulerHook(Callback):
    """A hook to update some hyper-parameters in optimizer, e.g., learning rate
    and momentum."""

    priority = 'LOW'

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Call step function for each scheduler after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        scheduler = trainer.lr_scheduler_configs
        param_schedulers = [scheduler] if not isinstance(scheduler, list) else scheduler
        if param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()

        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(scheduler, dict):
            for param_scheduler in scheduler.values():
                step(param_scheduler)
        else:
            raise TypeError(
                'trainer.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Call step function for each scheduler after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        scheduler = trainer
        param_schedulers = [scheduler] if not isinstance(scheduler, list) else scheduler
        if param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()

        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(scheduler, dict):
            for param_scheduler in scheduler.values():
                step(param_scheduler)
        else:
            raise TypeError(
                'trainer.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')


@HOOKS.register_module()
class YOLOv5ParamSchedulerHook(ParamSchedulerHook):
    """A hook to update learning rate and momentum in optimizer of YOLOv5."""
    priority = 9

    scheduler_maps = {'linear': linear_fn, 'cosine': cosine_fn}

    def __init__(self,
                 scheduler_type: str = 'linear',
                 lr_factor: float = 0.01,
                 max_epochs: int = 300,
                 warmup_epochs: int = 3,
                 warmup_bias_lr: float = 0.1,
                 warmup_momentum: float = 0.8,
                 warmup_mim_iter: int = 500,
                 **kwargs):

        assert scheduler_type in self.scheduler_maps

        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.warmup_mim_iter = warmup_mim_iter

        kwargs.update({'lr_factor': lr_factor, 'max_epochs': max_epochs})
        self.scheduler_fn = self.scheduler_maps[scheduler_type](**kwargs)

        self._warmup_end = False
        self._base_lr = None
        self._base_momentum = None


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        optimizer = trainer.optimizers[0]
        for group in optimizer.param_groups:
            # If the param is never be scheduled, record the current value
            # as the initial value.
            group.setdefault('initial_lr', group['lr'])
            group.setdefault('initial_momentum', group.get('momentum', -1))

        self._base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]
        self._base_momentum = [
            group['initial_momentum'] for group in optimizer.param_groups
        ]

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss) -> None:
        cur_iters = trainer.global_step
        cur_epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]

        # The minimum warmup is self.warmup_mim_iter
        warmup_total_iters = max(
            round(self.warmup_epochs * len(trainer.train_dataloader)),
            self.warmup_mim_iter)

        if cur_iters <= warmup_total_iters:
            xp = [0, warmup_total_iters]
            for group_idx, param in enumerate(optimizer.param_groups):
                if group_idx == 2:
                    # bias learning rate will be handled specially
                    yp = [
                        self.warmup_bias_lr,
                        self._base_lr[group_idx] * self.scheduler_fn(cur_epoch)
                    ]
                else:
                    yp = [
                        0.0,
                        self._base_lr[group_idx] * self.scheduler_fn(cur_epoch)
                    ]
                param['lr'] = np.interp(cur_iters, xp, yp)

                if 'momentum' in param:
                    param['momentum'] = np.interp(
                        cur_iters, xp,
                        [self.warmup_momentum, self._base_momentum[group_idx]])
        else:
            self._warmup_end = True

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._warmup_end:
            return

        cur_epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]
        for group_idx, param in enumerate(optimizer.param_groups):
            param['lr'] = self._base_lr[group_idx] * self.scheduler_fn(
                cur_epoch)

