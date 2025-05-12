import lightgbm as lgb
from seed import set_seed
from utils import preprocess_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

X, y, _, _ = preprocess_data("data/train_transaction.csv", "data/test_transaction.csv")

seed_list = [42, 56, 96, 100, 777]
seed_list = [7]
for seed in seed_list:
    set_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, f1s, accs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.01,
            n_estimators=5000,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='gpu_hist',
            use_label_encoder=False,
            verbosity=1,
            random_state=seed
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob > 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, prob))
        f1s.append(f1_score(y_val, pred))
        accs.append(accuracy_score(y_val, pred))

    print(f"[Seed {seed}] AUC: {np.mean(aucs):.4f}, F1: {np.mean(f1s):.4f}, ACC: {np.mean(accs):.4f}")