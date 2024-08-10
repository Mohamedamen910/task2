import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

raw_data = pd.read_csv('diabetes.csv')

print("Raw data:")
print(raw_data)

X = raw_data.drop('Outcome', axis=1)
y = raw_data['Outcome']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_preprocessed = preprocessor.fit_transform(X)

print("\nData after Imputation, Scaling, and Encoding:")
print(pd.DataFrame(X_preprocessed).head())

pca = PCA(n_components=0.95)

X_reduced = pca.fit_transform(X_preprocessed)

print("\nData after PCA (Feature Reduction):")
print(pd.DataFrame(X_reduced).head())

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

m = RandomForestClassifier(random_state=42)
m.fit(X_train, y_train)

accuracy = m.score(X_test, y_test)

print(f'\nModel Accuracy after preprocessing: {accuracy:.2f}')
