import os
import torch
import logging
from transformer_model.scripts.training.load_basis_model import load_moment_model
from transformer_model.scripts.config_transformer import CHECKPOINT_DIR

def load_final_transformer_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_moment_model()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    #logging.info(f"Loaded final model from: {checkpoint_path}")
    return model, device