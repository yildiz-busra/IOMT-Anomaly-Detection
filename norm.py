import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')

df = df.drop(columns=['Dir'])  
print("Before normalization:")
print(df.iloc[103:108])

label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

numerical_columns = df.drop(columns=['Label']).select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("After normalization:")
print(df.iloc[103:108])

