# load_basis_model.py
# Load and initialize the base MOMENT model before finetuning

import torch
import logging
from momentfm import MOMENTPipeline
from transformer_model.scripts.config_transformer import (
    FORECAST_HORIZON,
    FREEZE_ENCODER,
    FREEZE_EMBEDDER,
    FREEZE_HEAD,
    WEIGHT_DECAY,
    HEAD_DROPOUT,
    SEQ_LEN
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_moment_model():
    """
    Loads and configures the MOMENT model for forecasting.
    """
    logging.info("Loading MOMENT model...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': FORECAST_HORIZON,  # default = 1
            'head_dropout': HEAD_DROPOUT,          # default = 0.1
            'weight_decay': WEIGHT_DECAY,          # default = 0.0
            'freeze_encoder': FREEZE_ENCODER,      # default = True
            'freeze_embedder': FREEZE_EMBEDDER,    # default = True
            'freeze_head': FREEZE_HEAD             # default = False
        }
    )

    model.init()
    logging.info("Model initialized successfully.")
    return model


def print_trainable_params(model):
    """
    Logs all trainable (unfrozen) parameters of the model.
    """
    logging.info("Unfrozen parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"  {name}")


def test_dummy_forward(model):
    """
    Performs a dummy forward pass to verify the model runs without error.
    """
    logging.info("Running dummy forward pass with random tensors to see if model is running.")
    dummy_x = torch.randn(16, 1, SEQ_LEN)
    output = model(x_enc=dummy_x)
    logging.info("Dummy forward pass successful.")


if __name__ == "__main__":
    model = load_moment_model()
    print_trainable_params(model)
    test_dummy_forward(model)
