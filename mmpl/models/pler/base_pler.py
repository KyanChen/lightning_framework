import torch
import torch.nn as nn
from mmengine import OPTIM_WRAPPERS
from mmengine.optim import build_optim_wrapper, _ParamScheduler
import copy

from torchmetrics import MetricCollection

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
            for k, v in evaluator_cfg.items():
                metrics = []
                if isinstance(v, dict):
                    metric = METRICS.build(v)
                    metrics.append(metric)
                elif isinstance(v, list):
                    for metric_cfg in v:
                        metric = METRICS.build(metric_cfg)
                        metrics.append(metric)
                setattr(self, k, MetricCollection(metrics, prefix=k.split('_')[0]))

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

    def _set_train_module(self, mode=True, need_train_names: list=[]):
        self.training = mode
        import ipdb;
        ipdb.set_trace()
        for name, module in self.named_children():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.hyperparameters.get('optimizer'))
        base_lr = optimizer_cfg.get('lr')
        base_wd = optimizer_cfg.get('weight_decay', None)

        sub_models = optimizer_cfg.pop('sub_model', None)
        if sub_models is None:
            # optimizer_cfg['params'] = filter(lambda p: p.requires_grad, self.trainer.model.parameters())
            optimizer_cfg['params'] = self.parameters()
        else:
            if isinstance(sub_models, str):
                sub_models = {sub_models: {}}
            if isinstance(sub_models, list):
                sub_models = {x: {} for x in sub_models}
            assert isinstance(sub_models, dict), f'sub_models should be a dict, but got {type(sub_models)}'
            # import ipdb; ipdb.set_trace()
            # set training parameters and lr
            for sub_model_name, value in sub_models.items():
                sub_model_ = self.get_submodule(sub_model_name)
                # sub_model_ = self.trainer.strategy.model._forward_module.get_submodule(sub_model_name)
                if isinstance(sub_model_, torch.nn.Parameter):
                    # filter(lambda p: p.requires_grad, model.parameters())
                    # sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                else:
                    # import ipdb;ipdb.set_trace()
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, sub_model_.parameters())
                    # sub_models[sub_model_name]['params'] = sub_model_.parameters()
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
                for name, value in sub_models.items():
                    param_shape = [x.shape for x in value['params']]
                    print(f'{name}: {param_shape}', end=', ')
                print()

            optimizer_cfg['params'] = [value for key, value in sub_models.items()]

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        if self.local_rank == 0:
            print('查看优化器参数')
            import ipdb;ipdb.set_trace()
            for param_group in optimizer.param_groups:
                print(param_group.keys())
                # print(type(param_group))
                print([value.shape for value in param_group.values()])
                print('查看学习率: ', param_group['lr'])

        schedulers = copy.deepcopy(self.hyperparameters.get('param_scheduler', None))
        if schedulers is None:
            return [optimizer]
        param_schedulers = []
        total_step = self.trainer.estimated_stepping_batches
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
                            epoch_length=self.trainer.num_training_batches,
                        )
                    )
                )
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')

        return [optimizer], param_schedulers

    def lr_scheduler_step(self, scheduler, metric):
        pass

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, 'val_evaluator'):
            metrics = self.val_evaluator.compute()
            for k, v in metrics.items():
                v = v.view(-1)
                for i, data in enumerate(v):
                    self.log(f'{k.lower()}_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_evaluator.reset()

    def on_test_epoch_end(self) -> None:
        if hasattr(self, 'test_evaluator'):
            metrics = self.test_evaluator.compute()
            for k, v in metrics.items():
                v = v.view(-1)
                for i, data in enumerate(v):
                    self.log(f'{k.lower()}_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.test_evaluator.reset()

    def on_train_epoch_end(self) -> None:
        if hasattr(self, 'train_evaluator'):
            metrics = self.train_evaluator.compute()
            for k, v in metrics.items():
                v = v.view(-1)
                for i, data in enumerate(v):
                    self.log(f'{k.lower()}_{i}', data, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.train_evaluator.reset()