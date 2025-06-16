def hyperparameter():
    """
    Returns the complete KNN hyperparameter tuning code as a string.
    Usage:
    code = hyperparameter()
    print(code)  # To view the code
    exec(code)   # To execute the code
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


def aprior():
    """
    Returns Apriori association rule mining code as a string.
    Includes data loading options from multiple sources.
    Usage:
    code = aprior()
    print(code)  # To view the code
    exec(code)   # To execute the code
    """
    code = """
# Apriori Algorithm Implementation

# First install the package (remove ! if not in notebook)
!pip install apyori

from google.colab import files
import pandas as pd
from apyori import apriori

# Data loading options
def load_data(source='github'):
    if source == 'upload':
        uploaded = files.upload()
        return pd.read_csv(next(iter(uploaded.keys())))
    elif source == 'drive':
        from google.colab import drive
        drive.mount('/content/drive')
        return pd.read_csv('/content/drive/MyDrive/path_to_your_file.csv')
    elif source == 'github':
        url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Online%20Purchase.csv"
        return pd.read_csv(url)
    else:
        return pd.read_csv('purchase_data.csv', header=None)

# Load data
df = load_data('github')  # Change source as needed
print(df.to_string())

# Prepare records for Apriori
records = []
for i in range(0, len(df)):
    records.append([str(df.values[i, j]) for j in range(0, len(df.columns)) 
                   if pd.notna(df.values[i, j])])

# Generate association rules
association_rules = apriori(records, min_support=0.5, min_confidence=0.75)
association_results = list(association_rules)

# Print results
print("\\nAssociation Rules:")
for item in association_results:
    print("\\nRule:", item[0])
    print("Support:", item[1])
    print("Confidence:", item[2][0][2])
    print("Lift:", item[2][0][3])
"""
    return code