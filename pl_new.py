import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, TrainResult, EvalResult, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # , LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

import os
import time
from config import config
from utils import get_logger
from data import DataModule

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

    def forward(self, e1, r):
        pass

    # def _my_reduct_fx(self, all_train_loss):
    #     # "reducde somehow"
    #     result = None
    #     return resut

    def training_step(self, batch, batch_idx):
        loss = None
        result = TrainResult(minimize=loss,
                             early_stop_on=None,
                             checkpoint_on=None,
                             hiddens=None)
        result.log("train/loss",
                   loss,
                   prog_bar=False,
                   logger=True,
                   on_step=True,
                   on_epoch=False,
                   reduce_fx=torch.mean,
                   tbptt_reduce_fx=torch.mean,
                   tbptt_pad_token=0,
                   enable_graph=False,
                   sync_dist=False,
                   sync_dist_op='mean',
                   sync_dist_group=None)
        values = dict(train_acc=None, train_metric=None)
        result.log_dict(values,
                        prog_bar=False,
                        logger=True,
                        on_step=False,
                        on_epoch=True,
                        reduce_fx=torch.mean,
                        tbptt_reduce_fx=torch.mean,
                        tbptt_pad_token=0,
                        enable_graph=False,
                        sync_dist=False,
                        sync_dist_op='mean',
                        sync_dist_group=None)
        return result

    def validation_step(self, batch, batch_idx):
        acc = None
        result = EvalResult(early_stop_on=None,
                            checkpoint_on=acc,
                            hiddens=None)

        result.a_prediction = None

        result.log("val_acc",
                   acc,
                   prog_bar=False,
                   logger=True,
                   on_step=False,
                   on_epoch=True,
                   reduce_fx=torch.mean,
                   tbptt_reduce_fx=torch.mean,
                   tbptt_pad_token=0,
                   enable_graph=False,
                   sync_dist=False,
                   sync_dist_op='mean',
                   sync_dist_group=None)
        values = dict(val_metric=None)
        result.log_dict(values,
                        prog_bar=False,
                        logger=True,
                        on_step=False,
                        on_epoch=True,
                        reduce_fx=torch.mean,
                        tbptt_reduce_fx=torch.mean,
                        tbptt_pad_token=0,
                        enable_graph=False,
                        sync_dist=False,
                        sync_dist_op='mean',
                        sync_dist_group=None)
        return result

    def validation_epoch_end(self, outputs):
        all_a_predictions = outputs.a_prediction
        # "do something with the predictons from all validation_steps"

        return outputs

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({"val_acc":"test_acc", "val_metric":"test_metric")
        return result

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

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

    dm = DataModule(config, logger)
    dm.prepare_data()

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
