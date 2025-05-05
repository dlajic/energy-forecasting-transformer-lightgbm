# create_dataloaders.py

import logging
from torch.utils.data import DataLoader
from transformer_model.scripts.utils.informer_dataset_class import InformerDataset
from transformer_model.scripts.config_transformer import BATCH_SIZE, FORECAST_HORIZON
from momentfm.utils.utils import control_randomness

def create_dataloaders():
    logging.info("Setting random seeds...")
    control_randomness(seed=13)

    logging.info("Loading training dataset...")
    train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=FORECAST_HORIZON)
    logging.info("Train set loaded — Samples: %d | Features: %d", len(train_dataset), train_dataset.n_channels)

    logging.info("Loading test dataset...")
    test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=FORECAST_HORIZON)
    logging.info("Test set loaded — Samples: %d | Features: %d", len(test_dataset), test_dataset.n_channels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info("Dataloaders created successfully.")
    return train_loader, test_loader

if __name__ == "__main__":
    create_dataloaders()
