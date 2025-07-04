from lightgbm_model.scripts.utils import load_lightgbm_model
import numpy as np

def test_lightgbm_model_load():
    model = load_lightgbm_model()
    assert model is not None

def test_lightgbm_prediction():
    model = load_lightgbm_model()
    sample_input = np.random.rand(1, model.n_features_in_)  # Dummy Input
    prediction = model.predict(sample_input)
    assert prediction is not None
    assert len(prediction) == 1