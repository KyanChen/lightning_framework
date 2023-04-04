import torch
import torch.nn as nn
from mmengine import OPTIM_WRAPPERS
from mmengine.optim import build_optim_wrapper, _ParamScheduler
import copy
from mmpl.registry import MODELS
import lightning.pytorch as pl
from mmengine.registry import OPTIMIZERS, PARAM_SCHEDULERS
from mmengine.model import BaseModel


@MODELS.register_module()
class BasePLer(pl.LightningModule, BaseModel):
    def __init__(self, hyperparameters, *args, **kwargs):
        super().__init__()
        self.hyperparameters = hyperparameters

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.hyperparameters.get('optimizer'))
        optimizer_cfg['params'] = self.parameters()
        optimizer = OPTIMIZERS.build(optimizer_cfg)

        schedulers = copy.deepcopy(self.hyperparameters.get('param_scheduler'))
        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)
                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optimizer,
                            epoch_length=self.trainer.max_epochs)))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')

        return [optimizer], param_schedulers

    def lr_scheduler_step(self, scheduler, metric):
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
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')