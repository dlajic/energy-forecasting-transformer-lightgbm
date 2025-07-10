import logging
import os

import torch
from huggingface_hub import hf_hub_download

from transformer_model.scripts.config_transformer import CHECKPOINT_DIR
from transformer_model.scripts.training.load_basis_model import \
    load_moment_model

logging.basicConfig(level=logging.INFO)


# load model from checkpoint if available, else download it from hugging face
def load_real_transformer_model(device=None):  # ⬅️ Name geändert
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_moment_model()
    filename = "model_final.pth"
    local_path = os.path.join(CHECKPOINT_DIR, filename)

    if os.path.exists(local_path):
        checkpoint_path = local_path
        print("Loading model from local path...")
    else:
        print("Downloading model from Hugging Face Hub...")
        checkpoint_path = hf_hub_download(
            repo_id="dlaj/energy-forecasting-files",  # passe ggf. an
            filename=f"transformer_model/{filename}",
            repo_type="dataset",
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from: {checkpoint_path}")

    return model, device
