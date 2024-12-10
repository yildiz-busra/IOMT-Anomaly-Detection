import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')
df = df.drop(columns=['Dir', 'Attack Category'])

label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop(columns=['Label'])
y = df['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize model
model = XGBClassifier(random_state=42)

# Set up RFE
rfe = RFE(estimator=model, n_features_to_select=None, step=1)

# Fit RFE
rfe.fit(X_train, y_train)

# Results
print("Selected Features:", X.columns[rfe.support_])
print("Feature Ranking:", rfe.ranking_)

# Test performance
y_pred = rfe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
