# Generated from: CPL6.ipynb
# Converted at: 2026-06-01T10:54:28.937Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

import kagglehub

# Download dataset
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path:", path)

import os
import pandas as pd

file_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(file_path)

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Survival count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Sex")
plt.show()

# Survival by Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

print(df.columns)

cols_to_drop = ['Name', 'Ticket', 'Cabin']

df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Example (adapt based on your dataset)

# Fill missing values
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].mean())

if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Encode categorical
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(df.isnull().sum())

# Fill numeric columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical safely
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

import numpy as np

# Drop irrelevant columns safely
df.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore', inplace=True)

# Replace inf → NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values (numeric)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing values (categorical)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical (SAFE)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0)

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

# FINAL CHECK (must be zero)
print("Remaining NaN:\n", df.isnull().sum())

from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']

# Remove constant columns (important fix)
X = X.loc[:, X.nunique() > 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Split successful")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check for issues
print("NaN in X_train:", np.isnan(X_train).sum())
print("Inf in X_train:", np.isinf(X_train).sum())

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticModel(X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    preds = model(X_test).squeeze()
    preds = (preds > 0.5).float()

from sklearn.metrics import classification_report

y_test_cpu = y_test.cpu().numpy()
preds_cpu = preds.cpu().numpy()

print(classification_report(y_test_cpu, preds_cpu))