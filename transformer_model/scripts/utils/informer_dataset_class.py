# informer_dataset.py

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional
from transformer_model.scripts.config_transformer import SEQ_LEN, DATA_PATH

logging.basicConfig(level=logging.INFO)


class InformerDataset:
    def __init__(
        self,
        forecast_horizon: Optional[int],
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            'train' or 'test'.
        data_stride_len : int
            Stride length between time windows.
        task_name : str
            'forecasting' or 'imputation'.
        random_seed : int
            For reproducibility.
        """

        self.seq_len = SEQ_LEN
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = DATA_PATH
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        self._read_data()

    def _get_borders(self):
        train_ratio = 0.7
        n_train = int(self.length_timeseries_original * train_ratio)
        n_test = self.length_timeseries_original - n_train

        train_end = n_train
        test_start = train_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        #logging.info(f"Train range: 0 to {train_end}")
        #logging.info(f"Test range: {test_start} to {test_end}")

        return slice(0, train_end), slice(test_start, test_end)

    def _read_data(self):
        self.scaler = StandardScaler()

        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1  # exclude timestamp column

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()
        train_data = df[data_splits[0]]

        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

        #logging.info(f"{self.data_split.capitalize()} set loaded.")
        #logging.info(f"Time series length: {self.length_timeseries}")
        #logging.info(f"Number of features: {self.n_channels}")

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1
