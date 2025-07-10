from lightgbm_model.scripts.utils import load_lightgbm_model as real_model
from streamlit_simulation.utils.env import use_dummy


def load_lightgbm_model():
    if use_dummy():
        from streamlit_simulation.utils.dummy import DummyLightGBMModel

        return DummyLightGBMModel()
    else:
        return real_model()
