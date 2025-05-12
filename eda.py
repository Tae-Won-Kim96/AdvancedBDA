import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 경로 설정
folder_path = 'data/'
save_dir = 'results/'
train = pd.read_csv(os.path.join(folder_path, 'train.csv'))
eval_df = pd.read_csv(os.path.join(folder_path, 'evaluation.csv'))

# 기본 정보
print(f"✅ Train shape: {train.shape}")
print(f"✅ Evaluation shape: {eval_df.shape}")

# 라벨 분포
print("\n🎯 Label distribution (Train):")
print(train['isFraud'].value_counts())
print(f"Fraud rate: {100 * train['isFraud'].mean():.2f}%")

# 라벨 분포 시각화
plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=train)
plt.title("Label Distribution in Train Set")
plt.xlabel("isFraud")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(save_dir+"label_distribution.png")
plt.close()

# 결측치 분석
missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]
print(f"\n🚨 Columns with missing values: {len(missing)}")

plt.figure(figsize=(8, 6))
missing.head(30).plot(kind='barh')
plt.title("Top Missing Values in Train")
plt.xlabel("Proportion Missing")
plt.tight_layout()
plt.savefig(save_dir+"missing_values.png")
plt.close()

# 수치형 변수 분포 예시
num_feats = [col for col in train.columns if train[col].dtype in ['float64', 'int64'] and train[col].nunique() > 10]
example_cols = num_feats[:4]  # 예시로 4개만 시각화
train[example_cols].hist(bins=40, figsize=(12, 8))
plt.suptitle("Distribution of Sample Numerical Features")
plt.tight_layout()
plt.savefig(save_dir+"feature_distributions.png")
plt.close()

# 상관관계 분석 (상위 변수 20개 기준)
corr = train[num_feats].corr()
top_corr = corr.abs().sum().sort_values(ascending=False).head(20).index
plt.figure(figsize=(10, 8))
sns.heatmap(train[top_corr].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap (Top 20 Variables)")
plt.tight_layout()
plt.savefig(save_dir+"correlation_heatmap.png")
plt.close()

# 이상치 탐색 예시
plt.figure(figsize=(6, 4))
sns.boxplot(x='isFraud', y='TransactionAmt', data=train)
plt.title("Transaction Amount by Class")
plt.tight_layout()
plt.savefig(save_dir+"transaction_amount_by_class.png")
plt.close()

print("✅ EDA plots saved as PNGs.")
