import os

def load_data(path):
    with open(os.path.join(path), "r") as f:
        data = f.read()
    return data.split('\n')