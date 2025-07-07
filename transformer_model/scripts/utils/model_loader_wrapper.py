import os
from dotenv import load_dotenv

from transformer_model.scripts.utils.load_final_model import load_final_transformer_model as load_real_model

# Dummy fallback
try:
    from streamlit_simulation.dummy import DummyTransformerModel
except ImportError:
    DummyTransformerModel = None

load_dotenv()
USE_DUMMY = os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"

def load_transformer_model_only():
    if USE_DUMMY:
        if DummyTransformerModel is None:
            raise ImportError("DummyTransformerModel not available!")
        return DummyTransformerModel(), "cpu"
    else:
        return load_real_model()
