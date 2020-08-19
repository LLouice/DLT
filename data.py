from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger

    def prepare_data(self):
        self.logger.info(".... prepare_data ....")
        self.logger.info(".... prepare_data done! ....")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.ds_trn = None
            self.ds_val = None
        if stage == "test" or stage is None:
            self.ds_test = None

    def train_dataloader(self):
        self.dl_trn = DataLoader(self.ds_trn,
                                 batch_size=self.config.bs,
                                 shuffle=True,
                                 num_workers=self.config.nw0,
                                 pin_memory=True)
        return self.dl_trn

    def val_dataloader(self):
        self.dl_val = DataLoader(self.ds_val,
                                 batch_size=self.config.bs_dev,
                                 shuffle=False,
                                 num_workers=self.config.nw1,
                                 pin_memory=True)
        return self.dl_val

    def test_dataloader(self):
        self.dl_test = DataLoader(self.ds_test,
                                  batch_size=self.config.bs_dev,
                                  shuffle=False,
                                  num_workers=self.config.nw1,
                                  pin_memory=True)
        return self.dl_test
