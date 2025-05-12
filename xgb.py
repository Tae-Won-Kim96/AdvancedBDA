import lightgbm as lgb
from seed import set_seed
import pandas as pd
from utils import preprocess_data
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

save_dir = "results/"
X_train, y_train, X_eval, y_eval, _ = preprocess_data("data/train.csv", "data/evaluation.csv")

summary = {"Model": [], "Mean AUC": [], "Mean F1": [], "Mean ACC": []}

seed_list = [42, 56, 96, 100, 777]

aucs, f1s, accs = [], [], []
for seed in seed_list:
    set_seed(seed)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.01,
        n_estimators=4000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        tree_method='gpu_hist',
        use_label_encoder=False,
        early_stopping_rounds=100,
        verbosity=0,
        seed=seed
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)]
    )
    preds_proba = model.predict_proba(X_eval)[:, 1]
    preds = (preds_proba > 0.5).astype(int)

    aucs.append(roc_auc_score(y_eval, preds_proba))
    f1s.append(f1_score(y_eval, preds))
    accs.append(accuracy_score(y_eval, preds))

summary["Model"].append('XGBClassifier')
summary["Mean AUC"].append(np.mean(aucs))
summary["Mean F1"].append(np.mean(f1s))
summary["Mean ACC"].append(np.mean(accs))

results_df = pd.DataFrame(summary)
print("\n=== Evaluation Summary ===")
print(results_df)
results_df.to_csv(save_dir+"xgb_evaluation_summary.csv", index=False)
