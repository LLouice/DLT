import torch
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, hparams, logger):
        super(Net, self).__init__()
        self.hparams = hparams
        self._logger = logger

        # "init model"
        hp = self.hparams
        if hp.model == "Model0":
            # self.model =
            pass

    def prepare_data(self):
        self._logger.info(".... prepare_data ....")
        self._logger.info(".... prepare_data done! ....")

    def setup(self, stage):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        loss = None
        return {"loss": loss, "log": {"train/loss": loss.detach().item()}}

    def _dev_setp(self, batch, batch_idx, mode="val"):
        loss = None
        if mode == "val":
            pass
        else:
            pass
        return dict(log={"{}/loss".format(mode): loss.detach()})

    def _dev_epoch_end(self, outputs):
        log = dict()
        return log

    def validation_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx, mode="val")

    def validation_epoch_end(self, outputs):
        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["val/" + k] = v
        self._logger.info(tb_log)
        return dict(log=tb_log)

    def test_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx, mode="test")

    def test_epoch_end(self, outputs):
        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["test/" + k] = v
        self._logger.info(tb_log)
        return dict(log=tb_log)

    def configure_optimizers(self):
        if self.hparams.opt == "adamw":
            return torch.optim.AdamW(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.wd)
        elif self.hparams.opt == "adam":
            return torch.optim.Adam(self.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.wd)
        elif self.hparams.opt == "sgd":
            return torch.optim.SGD(self.parameters(),
                                   lr=self.hparams.lr,
                                   weight_decay=self.hparams.wd,
                                   momentum=0.9)