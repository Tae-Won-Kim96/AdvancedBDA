# evaluate.py
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from seed import set_seed
from utils import preprocess_data
import numpy as np


save_dir = "results/"
# Load pre-split data
X_train, y_train, X_eval, y_eval, _ = preprocess_data("data/train.csv", "data/evaluation.csv")

models = {
    "LightGBM": lambda seed: LGBMClassifier(
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
    ),
    "TabNet": lambda seed: TabNetClassifier(seed=seed, verbose=0)
}

summary = {"Model": [], "Mean AUC": [], "Mean F1": [], "Mean ACC": []}

seed_list = [42, 56, 96, 100, 777]

for model_name, model_fn in models.items():
    aucs, f1s, accs = [], [], []
    print(f"\nEvaluating: {model_name}")
    for seed in seed_list:
        set_seed(seed)
        model = model_fn(seed)

        if model_name == "TabNet":
            model.fit(X_train.values, y_train.values)
            preds_proba = model.predict_proba(X_eval.values)[:, 1]
            preds = model.predict(X_eval.values)
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                eval_metric="logloss",
                callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=100)]
            )
            preds_proba = model.predict_proba(X_eval)[:, 1]
            preds = (preds_proba > 0.5).astype(int)

        aucs.append(roc_auc_score(y_eval, preds_proba))
        f1s.append(f1_score(y_eval, preds))
        accs.append(accuracy_score(y_eval, preds))

    summary["Model"].append(model_name)
    summary["Mean AUC"].append(np.mean(aucs))
    summary["Mean F1"].append(np.mean(f1s))
    summary["Mean ACC"].append(np.mean(accs))

results_df = pd.DataFrame(summary)
print("\n=== Evaluation Summary ===")
print(results_df)
results_df.to_csv(save_dir+"evaluation_summary.csv", index=False)
