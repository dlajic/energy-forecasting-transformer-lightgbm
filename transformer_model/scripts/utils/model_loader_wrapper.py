# transformer_model/scripts/utils/model_loader_wrapper.py
import os
from dotenv import load_dotenv
from transformer_model.scripts.utils.load_final_model import load_real_transformer_model
from transformer_model.scripts.utils.informer_dataset_class import InformerDataset
from transformer_model.scripts.config_transformer import FORECAST_HORIZON

try:
    from streamlit_simulation.dummy import DummyTransformerModel, DummyDataset
except ImportError:
    DummyTransformerModel = None
    DummyDataset = None

load_dotenv()
USE_DUMMY = os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"

def load_final_transformer_model():
    if USE_DUMMY:
        if DummyTransformerModel is None:
            raise ImportError("DummyTransformerModel not available")
        return DummyTransformerModel(), "cpu"
    else:
        return load_real_transformer_model()

def load_model_and_dataset():
    model, device = load_final_transformer_model()

    if USE_DUMMY:
        if DummyDataset is None:
            raise ImportError("DummyDataset not available")
        dataset = DummyDataset(length=200)
    else:
        train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=FORECAST_HORIZON)
        test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=FORECAST_HORIZON)
        test_dataset.scaler = train_dataset.scaler
        dataset = test_dataset

    return model, dataset, device
