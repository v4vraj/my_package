def hyperparameter():
    """
    Returns the complete hyperparameter tuning code as a string.
    """
    code = """
# Dataset: Breast Cancer

# Importing all the required packages
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Step 1: Load the dataset
breast_cancer_df = pd.read_csv("/content/breast-cancer.csv")
print("\\nDataset Info:")
breast_cancer_df.info()
print("\\nDataset Description:")
print(breast_cancer_df.describe())

# Step 2: Drop unnecessary columns
breast_cancer_df.drop(['id'], axis=1, inplace=True)

# Step 3: Encode the target variable ('M' = 1, 'B' = 0)
breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M': 1, 'B': 0})

# Step 4: Separate features and target
X = breast_cancer_df.drop('diagnosis', axis=1)
y = breast_cancer_df['diagnosis']

# Step 5: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize KNN model
knn = KNeighborsClassifier()

# Step 8: Define hyperparameter grid for tuning
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Step 9: Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1)
print("\\nStarting Grid Search...")
grid_search.fit(X_train, y_train)

# Step 10: Evaluate the best model
print("\\nBest Hyperparameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Step 11: Print classification metrics
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
"""
    return code