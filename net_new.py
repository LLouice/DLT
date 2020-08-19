import torch
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams

        # "init model"
        hp = self.hparams
        if hp.model == "Model0":
            # self.model =
            pass

    def forward(self, x):
        pass

    # def _my_reduce_fx(self, all_train_loss):
    #     # "reducde somehow"
    #     result = None
    #     return result

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

        result.log("val/acc",
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
        values = {"val/metirc": None}
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
        result.rename_keys({
            "val/acc": "test/acc",
            "val/metric": "test/metric"
        })
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