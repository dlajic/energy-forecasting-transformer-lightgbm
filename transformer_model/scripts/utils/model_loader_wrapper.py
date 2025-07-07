# transformer_model/scripts/utils/model_loader_wrapper.py

import os
from dotenv import load_dotenv
from transformer_model.scripts.utils.load_final_model import load_real_transformer_model  # ⬅️ Achtung: Neuer Name!

try:
    from streamlit_simulation.dummy import DummyTransformerModel
except ImportError:
    DummyTransformerModel = None

load_dotenv()
USE_DUMMY = os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"

def load_final_transformer_model():  # ⬅️ Der Wrapper hat den alten "offiziellen" Namen
    if USE_DUMMY:
        if DummyTransformerModel is None:
            raise ImportError("DummyTransformerModel not available!")
        return DummyTransformerModel(), "cpu"
    else:
        return load_real_transformer_model()
