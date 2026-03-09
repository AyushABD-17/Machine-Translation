from keras.models import load_model

def save_model(model, path):
    model.save(path)
    print(f"Model saved to {path}")

def load_saved_model(path):
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model