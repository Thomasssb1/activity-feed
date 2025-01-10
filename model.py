import sys
import joblib


class Model:
    def __init__(self, path: str):
        self.path = path

    def save(self):
        joblib.dump(self, self.path)
        print(f"Model saved to {self.path}")

    def load(path: str):
        try:
            model = joblib.load(path)
            print(f"Model loaded from {path}")
            return model
        except FileNotFoundError:
            print(f"File {path} not found.")
            sys.exit()
