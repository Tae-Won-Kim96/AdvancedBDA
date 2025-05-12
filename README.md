# Fraud Detection using Deep Learning on Tabular Data

## Project Goal

The primary objective of this project is to train a deep learning model on the IEEE CIS Fraud Detection dataset (tabular) with the aim of achieving performance comparable to or surpassing that of established tree-based models.

## Data Structure

The dataset comprises approximately 600,000 data points, each characterized by 397 features. These features include a mix of categorical and continuous variables. A significant number of missing values are present, necessitating appropriate preprocessing techniques.

## Code Description

.
├── README.md

├── data

│   ├── train.csv

│   └── evaluation.csv

├── results

│   ├── EDA results...

│   └── Evaluation_summary.csv

├── eda.py

├── main.py

├── xgb.py

├── seed.py

├── utils.py

│   └── preprocess_data()


## Environment
tqdm
torch
lightgbm
pandas
matplotlib
seaborn
pytorch_tabnet
scikit-learn
numpy

```bash
pip install tqdm torch lightgbm pandas matplotlib seaborn pytorch-tabnet scikit-learn numpy
```  
## Script Execution Guide  
Data Placement: Ensure that train.csv and evaluation.csv are located in the data folder.  
Data Download: The tabular dataset can be downloaded from the following Google Drive link:  
https://drive.google.com/drive/folders/1KGttAE0JdvXctMTr6ruJsQPytHT3nseL?usp=drive_link  

Baseline Evaluation (LightGBM/TabNet): Run the `evaluation.py` script to train and evaluate the LightGBM and TabNet baseline models.  
XGBoost Evaluation: Execute the `xgb.py` script to train and evaluate the XGBoost model.  

## Experiments  
The following table summarizes the performance of different models:    

| Model        | Mean AUC | Mean F1 | Mean ACC |
|--------------|----------|---------|----------|
| LightGBM     | 0.9493   | 0.6639  | 0.9815   |
| XGBoost      | 0.9518   | 0.6613  | 0.9814   |
| TabNet (DL)  | 0.8888   | 0.4483  | 0.9501   |

## Baselines
G. Ke et al, "Lightgbm: A highly efficient gradient boosting decision tree." In NeurIPS, 2017.  
T. Chen et al, "Xgboost: A scalable tree boosting system." SIGKDD, 2016.  
S. Arik et al, "Tabnet: Attentive interpretable tabular learning." AAAI, 2021.  

## References
L. Grinsztajn et al, "Why do tree-based models still outperform deep learning on typical tabular data?" In NeurIPS, 2022.  
A. Kadra et al, “Well-tuned simple nets excel on tabular datasets.” In NeurIPS, 2021.  
