from typing import Any
from hydra.utils import instantiate

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

class RoadSignDataModule(LightningDataModule):
    def __init__(
        self,
        data_processor: Any,
        dataset_class: Dataset,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_processor = instantiate(data_processor)
        self.dataset_class_dict = dataset_class

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            dataset = instantiate(self.dataset_class_dict)
            dataset.setDataset(self.data_processor.get_data_list('train'))
            self.train_dataset = dataset
        print('train dataset size :',len(self.train_dataset))

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn = self.train_dataset.collate
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            dataset = instantiate(self.dataset_class_dict)
            dataset.setDataset(self.data_processor.get_data_list('val'))
            self.val_dataset = dataset
        print('val dataset size :', len(self.val_dataset))

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn = self.val_dataset.collate
        )
    
    def test_dataloader(self, testset_name = 'test1') -> DataLoader[Any]:
        if self.test_dataset is None:
            dataset = instantiate(self.dataset_class_dict)
            dataset.setDataset(self.data_processor.get_data_list(testset_name))
            self.test_dataset = dataset
        print('test dataset size :', len(self.test_dataset))

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn = self.test_dataset.collate
        )
