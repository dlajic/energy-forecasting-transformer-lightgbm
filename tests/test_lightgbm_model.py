from lightgbm_model.scripts.utils import load_lightgbm_model

def test_lightgbm_model_load():
    model = load_lightgbm_model()
    assert model is not None
