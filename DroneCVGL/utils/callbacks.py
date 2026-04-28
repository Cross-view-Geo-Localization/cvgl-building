import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from DroneCVGL.core.metrics.university import calc_sim, evaluate

class EvalAndSamplingCallback(Callback):
    """Handles evaluation and hard negative sampling callback after training one epoch"""
    def __init__(self, config, query_dataloader_test, reference_dataloader_test, reference_dataloader_train, train_dataset):
        super().__init__()
        self.config = config
        self.query_dl_test = query_dataloader_test
        self.reference_dl_test = reference_dataloader_test
        self.reference_dl_train = reference_dataloader_train
        self.train_dataset = train_dataset
        self.sim_dict = None

    def on_train_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch + 1

        # Evaluate
        if (epoch % self.config.eval.eval_every_n_epoch == 0) or epoch == self.config.training.epochs:

            print(f"\n{30*'-'}[ Evaluate Epoch: {epoch} ]{30*'-'}")

            metrics = evaluate(
                config=self.config,
                model=pl_module.model,
                query_loader=self.query_dl_test,
                reference_loader=self.reference_dl_test,
                ranks=[1,5,10],
                cleanup=True
            )

            pl_module.log("val/r1", metrics["r1"], prog_bar=True)
            pl_module.log("val/r5", metrics["r5"], prog_bar=True)
            pl_module.log("val/r10", metrics["r10"], prog_bar=True)
            pl_module.log("val/AP", metrics["AP"], prog_bar=True)

            if self.config.training.sim_sample:
                self.sim_dict = calc_sim(
                    config=self.config,
                    model=pl_module.model,
                    reference_dataloader=self.reference_dl_train,
                    step_size=1000,
                    cleanup=True
                )

        if self.config.training.custom_sampling:

            if self.config.training.sim_sample and self.sim_dict is not None:
                self.train_dataset.hard_negative_sampling_shuffle(
                    self.sim_dict,
                    neighbour_select=self.config.training.neighbour_select,
                    neighbour_range=self.config.training.neighbour_range
                )
            else:
                self.train_dataset.shuffle()
