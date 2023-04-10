import copy
import inspect
from typing import List, Union

import torch
import torch.nn as nn
import lightning
import torchmetrics

from mmengine.config import Config, ConfigDict
from mmpl.registry import METRICS


def register_pl_metrics() -> List[str]:
    """Register loggers in ``lightning.pytorch.loggers`` to the ``LOGGERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    pl_metrics = []
    for module_name in dir(torchmetrics):
        if module_name.startswith('__'):
            continue
        _metric = getattr(torchmetrics, module_name)
        if inspect.ismodule(_metric):
            for _metric_name in dir(_metric):
                if _metric_name.startswith('__'):
                    continue
                import ipdb;
                ipdb.set_trace()
                _metric = getattr(_metric, _metric_name)
                if inspect.isclass(_metric) and issubclass(_metric, torchmetrics.Metric):
                    METRICS.register_module(module=_metric)
                    pl_metrics.append(_metric_name)
            continue
        if inspect.isclass(_metric) and issubclass(_metric, torchmetrics.Metric):
            METRICS.register_module(module=_metric)
            pl_metrics.append(module_name)
    import ipdb; ipdb.set_trace()
    return pl_metrics


PL_METRICS = register_pl_metrics()

