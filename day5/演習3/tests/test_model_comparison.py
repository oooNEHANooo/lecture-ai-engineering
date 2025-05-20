import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# モデルをロードする関数
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# テストデータの準備（適宜変更）
X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")

# 過去バージョンとの性能比較
def test_model_performance():
    baseline_model = load_model("../models/titanic_model_baseline.pkl")
    new_model = load_model("../models/titanic_model.pkl")

    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_new = new_model.predict(X_test)

    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    acc_new = accuracy_score(y_test, y_pred_new)

    assert acc_new >= acc_baseline, f"🚨 モデルの性能が低下しています！ (baseline={acc_baseline}, new={acc_new})"
