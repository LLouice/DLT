import sys
import os
import time

import torch
from pytorch_lightning import Trainer, TrainResult, EvalResult, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # , LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

from config import config
from utils import get_logger
from data import DataModule
from net_new import Net

logger = get_logger(config)

torch.set_printoptions(precision=5)
seed_everything(42)


class TimeCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.tc = time.perf_counter()

    def on_epoch_end(self, trainer, pl_module):
        logger.debug(f"Elapsed Time {time.perf_counter()-self.tc:.4f}")
        self.tc = time.perf_counter()


def main(config):
    logger.info(" ".join(sys.argv))

    dm = DataModule(config, logger)
    dm.prepare_data()

    logger.info(f"{'='*42} \n\t {config} \n\t {'='*42}")

    def train():
        model = Net(config, logger)

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
            dm.setup("fit")
            lr_finder = trainer.lr_find(model, dm.dl_trn, dm.dl_val)
            # "Inspect results"
            # fig = lr_finder.plot(); fig.show(); fig.savefig("lr_finder.png")
            suggested_lr = lr_finder.suggestion()
            logger.info("suggested_lr: ", suggested_lr)
        else:
            dm.setup("fit")
            trainer.fit(model, datamodule=dm)
            logger.success("training finish!")
            dm.setup("test")
            trainer.test(datamodule=dm)

    train()


if __name__ == "__main__":
    main(config)
