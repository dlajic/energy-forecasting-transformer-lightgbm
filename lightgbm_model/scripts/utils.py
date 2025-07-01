import pickle
import os

MODEL_PATH = os.path.join("lightgbm_model", "model", "lightgbm_final_model.pkl")

def load_lightgbm_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)