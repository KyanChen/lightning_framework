import torch
import torch.nn as nn
from mmengine import OPTIM_WRAPPERS
from mmengine.optim import build_optim_wrapper, _ParamScheduler
import copy
from mmpl.registry import MODELS, METRICS
import lightning.pytorch as pl
from mmengine.registry import OPTIMIZERS, PARAM_SCHEDULERS
from mmengine.model import BaseModel


@MODELS.register_module()
class BasePLer(pl.LightningModule, BaseModel):
    def __init__(self, hyperparameters, data_preprocessor=None, *args, **kwargs):
        super().__init__()
        self.hyperparameters = hyperparameters
        if data_preprocessor is not None:
            if isinstance(data_preprocessor, nn.Module):
                self.data_preprocessor = data_preprocessor
            elif isinstance(data_preprocessor, dict):
                self.data_preprocessor = MODELS.build(data_preprocessor)
            else:
                raise TypeError('data_preprocessor should be a `dict` or '
                                f'`nn.Module` instance, but got '
                                f'{type(data_preprocessor)}')

        evaluator_cfg = copy.deepcopy(self.hyperparameters.get('evaluator', None))
        if evaluator_cfg is not None:
            self.evaluator = METRICS.build(evaluator_cfg)
            # self.train_metrics = metrics.clone(prefix='train_')

    def _set_grad(self, need_train_names: list=[], noneed_train_names: list=[]):

        for name, param in self.named_parameters():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            for noneed_train_name in noneed_train_names:
                if noneed_train_name in name:
                    flag = False
            param.requires_grad_(flag)

        not_specific_names = []
        for name, param in self.named_parameters():
            flag_find = False
            for specific_name in need_train_names + noneed_train_names:
                if specific_name in name:
                    flag_find = True
            if not flag_find:
                not_specific_names.append(name)

        if self.local_rank == 0:
            not_specific_names = [x.split('.')[0] for x in not_specific_names]
            not_specific_names = set(not_specific_names)
            print(f"Turning off gradients for names: {noneed_train_names}")
            print(f"Turning on gradients for names: {need_train_names}")
            print(f"Turning off gradients for not specific names: {not_specific_names}")

    def _set_train_module(self, mode=True):
        self.training = mode
        for name, module in self.named_children():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.hyperparameters.get('optimizer'))
        base_lr = optimizer_cfg.pop('lr')
        base_wd = optimizer_cfg.pop('weight_decay', None)

        sub_models = optimizer_cfg.pop('sub_model', None)
        if sub_models is None:
            optimizer_cfg['params'] = self.parameters()
        else:
            if isinstance(sub_models, str):
                sub_models = {sub_models: {}}
            if isinstance(sub_models, list):
                sub_models = {x: {} for x in sub_models}

            # set training parameters and lr
            for sub_model_name, value in sub_models.items():
                sub_model_ = self.get_submodule(sub_model_name)
                if isinstance(sub_model_, torch.nn.Parameter):
                    # filter(lambda p: p.requires_grad, model.parameters())
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                else:
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, sub_model_.parameters())
                lr_mult = value.pop('lr_mult', 1.)
                sub_models[sub_model_name]['lr'] = base_lr * lr_mult
                if base_wd is not None:
                    decay_mult = value.pop('decay_mult', 1.)
                    sub_models[sub_model_name]['weight_decay'] = base_wd * decay_mult
                else:
                    raise ModuleNotFoundError(f'{sub_model_name} not in model')

            if self.local_rank == 0:
                print('All sub models:')
                for name, module in self.named_children():
                    print(name, end=', ')
                print()
                print('Needed train models:')
                for needed_train_sub_model in sub_models.keys():
                    print(needed_train_sub_model, end=', ')
                print()

            optimizer_cfg['params'] = [value for key, value in sub_models.items()]

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

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, 'evaluator'):
            metrics = self.evaluator.compute()
            for i, data in enumerate(metrics):
                self.log(f'metric_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.evaluator.reset()

    def on_test_epoch_end(self) -> None:
        if hasattr(self, 'evaluator'):
            metrics = self.evaluator.compute()
            for i, data in enumerate(metrics):
                self.log(f'metric_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.evaluator.reset()
