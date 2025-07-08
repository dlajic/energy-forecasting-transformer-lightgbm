# streamlit_simulation/dummy.py
import numpy as np
import torch

class DummyDataset:
    def __init__(self, length=100):
        self.data = np.zeros((length, 10))  # Dummydaten
        self.scaler = DummyScaler()
        self.n_channels = 1
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        timeseries = np.zeros((48, 1))  # (SEQ_LEN, Channels)
        target = np.zeros((1, 1))       # Forecast target
        mask = np.ones((48,))           # Dummy-Maske
        return timeseries, target, mask

class DummyScaler:
    def inverse_transform(self, x):
        return x  # keine Skalierung nötig

class DummyOutput:
    def __init__(self, forecast_shape):
        # gib einen echten Tensor zurück, wie vom echten Modell erwartet
        self.forecast = torch.tensor(np.full(forecast_shape, 42.0), dtype=torch.float32)

class DummyTransformerModel:
    def __call__(self, x_enc, input_mask):
        batch_size, seq_len, channels = x_enc.shape
        forecast_shape = (batch_size, 1, channels)
        return DummyOutput(forecast_shape)

class DummyLightGBMModel:
    def predict(self, X):
        return np.zeros(len(X))  # ← gibt jetzt np.ndarray zurück
