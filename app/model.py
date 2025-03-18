import os
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = "data/iris.pkl"

def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = DecisionTreeClassifier()
    model.fit(X,y)

    os.makedirs("data", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    if not os.path.exists(MODEL_PATH)
        train_and_save_model()
    return joblib.load(MODEL_PATH)