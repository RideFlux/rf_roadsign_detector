import rootutils
import torch
torch.set_float32_matmul_precision('high')

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger

from utils import (
    RankedLogger,
    instantiate_callbacks,
)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger: List[Logger] = TensorBoardLogger("tb_logs", name="pointpillars-3d")

    datamodule : LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    model : LightningModule = hydra.utils.instantiate(cfg.model.roadsign_detector.module)

    trainer.fit(model=model, 
                train_dataloaders=datamodule.train_dataloader(), 
                val_dataloaders=datamodule.val_dataloader())

    return

if __name__ == "__main__":
    main()
