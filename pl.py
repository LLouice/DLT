import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # , LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

import os
import time
from config import config
from utils import get_logger

logger = get_logger(config)

torch.set_printoptions(precision=5)
seed_everything(42)


class TimeCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.tc = time.perf_counter()

    def on_epoch_end(self, trainer, pl_module):
        logger.debug(f"Elapsed Time {time.perf_counter()-self.tc:.4f}")
        self.tc = time.perf_counter()


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams

        # "init model"
        hp = self.hparams
        if hp.model == "Model0":
            # self.model =
            pass

    def prepare_data(self):
        logger.info(".... prepare_data ....")
        logger.info(".... prepare_data done! ....")

    def setup(self, stage):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def forward(self, e1, r):
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
        logger.info(tb_log)
        return dict(log=tb_log)

    def test_step(self, batch, batch_idx):
        return self._dev_setp(batch, batch_idx, mode="test")

    def test_epoch_end(self, outputs):
        res = self._dev_epoch_end(outputs)
        tb_log = {}
        for k, v in res.items():
            tb_log["test/" + k] = v
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


def main(config):
    logger.info(f"{'='*42} \n\t {config} \n\t {'='*42}")

    def train():
        model = Net(config)

        # "checkpoint"
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                os.getcwd(),
                "runs/ckpts/" + config.model + "/{epoch}-{val_acc:.4f}"),
            save_top_k=1,
            verbose=True,
            monitor='val_acc',
            mode='max',
        )

        # "tensorboard logger"
        tb_logger = TensorBoardLogger(
            "runs/logs",
            name=config.model,
        )

        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=5,
                                       strict=False,
                                       verbose=False,
                                       mode='max')
        if config.tpu:
            gpus = None
        else:
            gpus = config.gpus if not config.nogpu else None

        distributed_backend = None
        if len(config.gpus.split(",")) > 1 or config.gpus == "-1":
            distributed_backend = "ddp"

        trainer = Trainer(
            fast_dev_run=config.dbg,
            gpus=gpus,
            tpu_cores=8 if config.tpu else None,
            max_epochs=config.epos,
            check_val_every_n_epoch=config.check_val,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stopping,
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=config.pb_rate,
            distributed_backend=distributed_backend,
            logger=tb_logger,
            # callbacks=[TimeCallback()],
            # resume_from_checkpoint="ckpts/foo.ckpt"
        )

        if config.lr_find:
            lr_finder = trainer.lr_find(model)
            # "Inspect results"
            # fig = lr_finder.plot(); fig.show(); fig.savefig("lr_finder.png")
            suggested_lr = lr_finder.suggestion()
            logger.info("suggested_lr: ", suggested_lr)
        else:
            trainer.fit(model)
            logger.success("training finish!")
            trainer.test()

    train()


if __name__ == "__main__":
    main(config)
