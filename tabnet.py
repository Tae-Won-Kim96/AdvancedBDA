from pytorch_tabnet.tab_model import TabNetClassifier
from seed import set_seed
from utils import preprocess_data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

X, y, _, _ = preprocess_data("data/train_transaction.csv", "data/test_transaction.csv")

seed_list = [42, 56, 96, 100, 777]
seed_list = [7]
for seed in seed_list:
    set_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, f1s, accs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

        model = TabNetClassifier(seed=seed, verbose=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=["auc"], patience=20)

        preds = model.predict(X_val)
        preds_proba = model.predict_proba(X_val)[:, 1]
        
        aucs.append(roc_auc_score(y_val, preds_proba))
        f1s.append(f1_score(y_val, preds))
        accs.append(accuracy_score(y_val, preds))

    print(f"[TabNet Seed {seed}] AUC: {np.mean(aucs):.4f}, F1: {np.mean(f1s):.4f}, ACC: {np.mean(accs):.4f}")