import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

def preprocess_data(train_path, test_path):
    # Load full data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    label_encoders = {}

    # Label Encoding with shared encoder across train/test
    for col in tqdm(categorical_cols):
        combined = pd.concat([train[col], test[col]], axis=0).fillna('unknown').astype(str)
        le = LabelEncoder()
        le.fit(combined)
        train[col] = le.transform(train[col].fillna('unknown').astype(str))
        test[col] = le.transform(test[col].fillna('unknown').astype(str))
        label_encoders[col] = le

    features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
    features = [col for col in features if train[col].isnull().mean() < 0.4]
    features = [col for col in features if col not in ['isFraud', 'TransactionID']]
    features = list(set(features + categorical_cols))
    print(f"Features: {features}")

    # Fill missing values
    for col in tqdm(features):
        if train[col].dtype in ['float64', 'int64']:
            median = train[col].median()
            train[col] = train[col].fillna(median)
            test[col] = test[col].fillna(median)
        else:
            train[col] = train[col].fillna('unknown')
            test[col] = test[col].fillna('unknown')

    # Shared scaler
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    return train[features], train['isFraud'], test[features], test['isFraud'], features
