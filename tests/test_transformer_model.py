from transformer_model.scripts.training.load_basis_model import load_moment_model

print("🚨 Transformer test file loaded")

def test_load_moment_model():
    model = load_moment_model()
    assert model is not None
    assert hasattr(model, "forward")  # oder eine spezifische Methode, die dein Modell immer hat