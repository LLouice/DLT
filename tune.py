import sys
import os
import time

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # , LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities.parsing import AttributeDict

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.utils import pin_in_object_store, get_pinned_object
from pytorch_lightning.utilities.cloud_io import load as pl_load
from functools import partial

from config import config
from utils import get_logger
from net import Net

logger = get_logger(config)

torch.set_printoptions(precision=5)
seed_everything(42)

os.environ["CUDA_VISIBLE_DEVICES"] = config.tune_gpus

ray.init(
    # object_store_memory=5 * 10e9,
    num_cpus=config.tune_num_cpus,
    num_gpus=config.tune_num_gpus,
    # resources={
    #     'Resource1': 2,
    #     'Resource2': 3
    # },
    local_mode=False,
    dashboard_host="0.0.0.0",
    # webui_host="0.0.0.0"
)


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(vacc=trainer.callback_metrics["val/acc"].item())

    def on_test_end(self, trainer, pl_module):
        tune.report(tacc=trainer.callback_metrics["test/acc"].item())


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))


class TimeCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.tc = time.perf_counter()

    def on_epoch_end(self, trainer, pl_module):
        logger.debug(f"Elapsed Time {time.perf_counter()-self.tc:.4f}")
        self.tc = time.perf_counter()


def main(config):
    logger.info(f"{'='*42} \n\t {config} \n\t {'='*42}")

    dl_trn, dl_val, dl_test = None, None, None

    dl_trn_id = pin_in_object_store(dl_trn)
    logger.info("pin trn done!")
    dl_val_id = pin_in_object_store(dl_val)
    logger.info("pin val done!")
    dl_test_id = pin_in_object_store(dl_test)
    logger.info("pin test done!")
    logger.info("pin object done!")

    def train(
        config,
        checkpoint_dir=None,
    ):
        # "get pin data"
        dl_trn = get_pinned_object(dl_trn_id)
        dl_val = get_pinned_object(dl_val_id)
        dl_test = get_pinned_object(dl_test_id)

        # "tensorboard logger"
        tb_logger = TensorBoardLogger("runs/logs", name="", version=".")

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

        if config.tune_schd == "asha":
            callbacks = [TuneReportCallback()]
        elif config.tune_schd == "pbt":
            callbacks = [CheckpointCallback(), TuneReportCallback()]
        trainer = Trainer(
            fast_dev_run=config.dbg,
            gpus=gpus,
            tpu_cores=8 if config.tpu else None,
            max_epochs=config.epos,
            check_val_every_n_epoch=config.check_val,
            # checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stopping,
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=config.pb_rate,
            distributed_backend=distributed_backend,
            logger=tb_logger,
            callbacks=callbacks,
        )

        # "make model"
        if checkpoint_dir:
            # "Currently, this leads to errors:"
            try:
                model = Net.load_from_checkpoint(
                    os.path.join(checkpoint_dir, "checkpoint"))
                logger.warning("can load!, not need workaroud")
            except Exception as e:
                logger.warning(e)
                # "Workaround:"
                ckpt = pl_load(os.path.join(checkpoint_dir, "checkpoint"),
                               map_location=lambda storage, loc: storage)
                model = Net._load_model_state(ckpt, config=config)
                logger.warning("use workaroud")
            trainer.current_epoch = ckpt["epoch"]
        else:
            model = Net(config, logger)

        trainer.fit(model, dl_trn, dl_val)
        logger.info("training finish!")
        trainer.test(test_dataloaders=dl_test)

    # reporter = CLIReporter(
    #     parameter_columns=["lr", "wd", "model"],
    #     metric_columns=["loss", "vacc", "training_iteration"])

    def tune(config, num_samples=8):
        if config.tune_schd == "asha":
            scheduler = ASHAScheduler(metric="vacc",
                                      mode="max",
                                      max_t=config.epos,
                                      grace_period=1,
                                      reduction_factor=2)

            # "tune config"
            config.__dict__.update({
                # "lr": tune.grid_search([0.004, 0.0015]),
                "lr":
                tune.loguniform(1e-4, 5e-3),
                # "bs": tune.choice([64, 128, 256]),
                # "opt": tune.choice(["adam", "adamw"]),
                "wd":
                tune.grid_search([0., 1e-4]),
                "model":
                tune.grid_search(["Model0", "Model1"]),
            })
        elif config.tune_schd == "pbt":
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="vacc",
                mode="max",
                perturbation_interval=4,
                hyperparam_mutations={
                    "lr": lambda: tune.loguniform(5e-5, 1e-1).func(None),
                })
            # "tune config"
            config.__dict__.update({
                "lr":
                1e-3,
                # "bs": tune.choice([64, 128, 256]),
                # "opt": tune.choice(["adam", "adamw"]),
                # "wd": tune.choice([0., 1e-4, 3e-4, 1e-5]),
                # "wd": tune.grid_search([0., 1e-4]),
                "wd":
                1e-4,
                "model":
                tune.grid_search(["Model0", "Model1"]),
            })

        analysis = tune.run(
            train,
            resources_per_trial={
                "cpu": config.tune_per_cpu,
                "gpu": config.tune_per_gpu,
            },
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            # progress_reporter=reporter,
            name=config.tune_name,
            local_dir="runs/tune_results")
        logger.info("Best config: {}".format(
            analysis.get_best_config(metric="tacc")))

    tune(config, num_samples=config.tune_num_samples)


if __name__ == "__main__":
    main(config)
