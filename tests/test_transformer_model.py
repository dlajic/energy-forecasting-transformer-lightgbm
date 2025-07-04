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


#def test_transformer_prediction_with_dataloader():
#    # Modell laden
#    model, device = load_final_transformer_model()
#
#    # Dataset + Dataloader
#    dataset = InformerDataset(data_split="test", forecast_horizon=FORECAST_HORIZON, task_name="forecasting")
#    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#    # Erstes Batch holen
#    for timeseries, forecast, input_mask in dataloader:
#        timeseries = timeseries.to(device).float()
#        input_mask = input_mask.to(device)
#        break  # Nur ein Batch
#
#    # Vorwärtslauf
#    with torch.no_grad():
#        output = model(x_enc=timeseries, input_mask=input_mask)
#
#    # Assertions
#    assert output.forecast is not None
#    assert output.forecast.shape[0] == BATCH_SIZE


def test_transformer_real_prediction():
    # Modell & Gerät laden
    model, device = load_final_transformer_model()

    # Echtes Test-Dataset laden
    dataset = InformerDataset(data_split="test", forecast_horizon=FORECAST_HORIZON)

    # Erstes Sample holen
    timeseries, _, input_mask = dataset[0]  # [C, T], [C, T_pred], [T]
    
    # In richtige Tensor-Form bringen
    x = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0).to(device)        # [1, C, T]
    mask = torch.tensor(input_mask, dtype=torch.bool).unsqueeze(0).to(device)        # [1, T]

    # Prediction
    with torch.no_grad():
        output = model(x_enc=x, input_mask=mask)

    # Assertions
    assert output is not None
    assert hasattr(output, "forecast")
    assert isinstance(output.forecast, torch.Tensor)
    assert output.forecast.shape[0] == 1  # Batch size