from pathlib import Path
from abc import ABC, abstractclassmethod
import torch
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(cfg):

    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = str(Path(cfg.config).parent) + f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    version_name = Path(cfg.config).name[:-5]
    
    
    #---->TensorBoard
    if cfg.stage != 'test':
        cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'
        tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_name),
                                                name = version_name, version = f'fold{cfg.Data.fold}',
                                                log_graph = True, default_hp_metric = False)
        #---->CSV
        csv_logger = pl_loggers.CSVLogger(log_path+str(log_name),
                                        name = version_name, version = f'fold{cfg.Data.fold}', )
    else:  
        cfg.log_path = Path(log_path) / log_name / version_name / f'test'
        tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_name),
                                                name = version_name, version = f'test',
                                                log_graph = True, default_hp_metric = False)
        #---->CSV
        csv_logger = pl_loggers.CSVLogger(log_path+str(log_name),
                                        name = version_name, version = f'test', )
                                    
    
    print(f'---->Log dir: {cfg.log_path}')

    # return tb_logger
    return [tb_logger, csv_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )

    Mycallbacks.append(early_stop_callback)
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description='green_yellow',
            progress_bar='green1',
            progress_bar_finished='green1',
            batch_progress='green_yellow',
            time='grey82',
            processing_speed='grey82',
            metrics='grey82'

        )
    )
    Mycallbacks.append(progress_bar)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_auc',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_auc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=0) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(y.shape[0])])
    loss = - torch.sum(x_log) / y.shape[0]
    return loss

#-----> convert labels for task
label_map = {
    'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'no_other': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': None},
    'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},

}
def convert_labels_for_task(task, label):

    return label_map[task][label]


#-----> KFOLD LOOP

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)