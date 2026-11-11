import asyncio
import datetime

from torch.utils.data import DataLoader
from .ICLR_dataset import ICLRDataset
from .NeurIPS_dataset import NeurIPSDataset


DATASETS = {
    "ICLR": ICLRDataset,
    "NeurIPS": NeurIPSDataset
}


def custom_collate_fn(batch):
    """
    Custom collate function for batching complex data structures.
    """
    if isinstance(batch[0], dict):
        return {key: [d[key] for d in batch] for key in batch[0]}
    return batch


async def load_dataset(dataset_dir: str, batch_size: int, shuffle: bool, num_workers: int, **kwargs) -> DataLoader:
    """
    Load a dataset and return the DataLoader.

    Args:
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.
                
    Returns:
        dataloader: A DataLoader for the dataset.
    """


    # check dataset source
    if "ICLR" in dataset_dir:
        dataset_source = "ICLR"
    elif "NeurIPS" in dataset_dir:
        dataset_source = "NeurIPS"
    else:
        raise ValueError(f"Dataset venue not found in {dataset_dir}")


    print("="*50)
    print(f"Dataset Loading - {datetime.datetime.now()}")

    # with filtering
    dataset = await DATASETS[dataset_source].create(dataset_dir, **kwargs)

    print(f"Dataset - {dataset_source} Loaded - {datetime.datetime.now()}")
    print("="*50)

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=custom_collate_fn)

    return dataloader
