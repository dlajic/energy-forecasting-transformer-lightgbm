from transformer_model.scripts.training.load_basis_model import load_moment_model
from transformer_model.scripts.utils.load_final_model import load_final_transformer_model
from transformer_model.scripts.utils.informer_dataset_class import InformerDataset
import torch
from transformer_model.scripts.config_transformer import BATCH_SIZE, FORECAST_HORIZON
from torch.utils.data import DataLoader

print("🚨 Transformer test file loaded")

def test_load_moment_model():
    model = load_moment_model()
    assert model is not None
    assert hasattr(model, "forward")  # oder eine spezifische Methode, die dein Modell immer hat

def test_load_final_model():
    model, device = load_final_transformer_model()
    assert model is not None
    assert model.training is False  # eval() sollte gesetzt sein
    assert hasattr(model, "forward")


def test_transformer_prediction_with_dataloader():
    # Modell laden
    model, device = load_final_transformer_model()

    # Dataset + Dataloader
    dataset = InformerDataset(data_split="test", forecast_horizon=FORECAST_HORIZON, task_name="forecasting")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Erstes Batch holen
    for timeseries, forecast, input_mask in dataloader:
        timeseries = timeseries.to(device).float()
        input_mask = input_mask.to(device)
        break  # Nur ein Batch

    # Vorwärtslauf
    with torch.no_grad():
        output = model(x_enc=timeseries, input_mask=input_mask)

    # Assertions
    assert output.forecast is not None
    assert output.forecast.shape[0] == BATCH_SIZE
