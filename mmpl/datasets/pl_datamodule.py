from mmpl.registry import DATASETS
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .builder import build_dataset


@DATASETS.register_module()
class PLDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 predict_loader=None,
                 **kwargs
                 ):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.predict_loader = predict_loader
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            dataset_cfg = self.train_loader.pop('dataset')
            self.train_dataset = build_dataset(dataset_cfg)
        if stage == "val":
            if self.val_loader is not None:
                dataset_cfg = self.val_loader.pop('dataset')
                self.val_dataset = build_dataset(dataset_cfg)
        if stage == "test":
            if self.test_loader is not None:
                dataset_cfg = self.test_loader.pop('dataset')
                self.test_dataset = build_dataset(dataset_cfg)
        if stage == "predict":
            if self.predict_loader is not None:
                dataset_cfg = self.predict_loader.pop('dataset')
                self.predict_dataset = build_dataset(dataset_cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_loader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.test_loader)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, **self.predict_loader)

    # collate_fn_cfg = dataloader_cfg.pop('collate_fn',
    #                                     dict(type='pseudo_collate'))
    # collate_fn_type = collate_fn_cfg.pop('type')
    # collate_fn = FUNCTIONS.get(collate_fn_type)
    # collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    # data_loader = DataLoader(
    #     dataset=dataset,
    #     sampler=sampler if batch_sampler is None else None,
    #     batch_sampler=batch_sampler,
    #     collate_fn=collate_fn,
    #     worker_init_fn=init_fn,
    #     **dataloader_cfg)
