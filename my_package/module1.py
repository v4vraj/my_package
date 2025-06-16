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


def decisiontree():
    """
    Returns Decision Tree classification code as a string.
    Usage:
    code = DT()
    print(code)  # To view the code
    exec(code)   # To execute the code
    """
    code = """
# Decision Tree Implementation
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

# Load dataset
df = pd.read_csv("golf_df.csv")
print("Dataset Preview:")
print(df.head())

# Prepare features and target
X = df.drop('Play', axis=1)
y = df['Play']

# Encode categorical features
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Visualize tree
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\\nModel Evaluation:")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, pos_label='yes'))
print("Recall score:", recall_score(y_test, y_pred, pos_label='yes'))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))
"""
    return code

def naivebayes():
    """
    Returns Naive Bayes classification code as a string.
    Usage:
    code = NB()
    print(code)  # To view the code
    exec(code)   # To execute the code
    """
    code = """
# Naive Bayes Implementation
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Load dataset (assuming df is already loaded)
print("\\nDataset Preview:")
print(df.head())

# Prepare features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Handle missing values
X['Gender'] = X['Gender'].fillna(X['Gender'].mode()[0])
X['Country'] = X['Country'].fillna(X['Country'].mode()[0])
X['Education'] = X['Education'].fillna(X['Education'].mode()[0])
X['Salary'] = X['Salary'].fillna(X['Salary'].mean())

# Encode categorical variables
X_encoded = pd.get_dummies(X).astype(int)
print("\\nEncoded Features:")
print(X_encoded.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\\nModel Evaluation:")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# Return model and results if needed
# return model, y_pred, accuracy_score(y_test, y_pred)
"""
    return code

def plots():
    """
    Returns matplotlib visualization examples as executable code string.
    Includes 9 common plot types with proper formatting and styling.
    Usage:
    code = plots()
    print(code)  # To view the code
    exec(code)   # To execute and display plots
    """
    code = """
# Comprehensive Visualization Examples
import matplotlib.pyplot as plt
import numpy as np

# Set global style
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Basic Line Plot
plt.figure(1)
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, marker='o', linestyle='--', color='blue')
plt.title("Basic Line Plot", fontsize=14)
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.grid(True)
plt.show()

# 2. Scatter Plot
plt.figure(2)
x = [5, 7, 8, 7]
y = [99, 86, 87, 88]
plt.scatter(x, y, color='red', s=100, alpha=0.7)
plt.title("Scatter Plot", fontsize=14)
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.show()

# 3. Bar Chart
plt.figure(3)
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values, color='orange', edgecolor='black')
plt.title("Bar Chart", fontsize=14)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.show()

# 4. Horizontal Bar Chart
plt.figure(4)
plt.barh(categories, values, color='green', edgecolor='black')
plt.title("Horizontal Bar Chart", fontsize=14)
plt.xlabel("Values", fontsize=12)
plt.ylabel("Category", fontsize=12)
plt.show()

# 5. Histogram
plt.figure(5)
data = np.random.randn(1000)
plt.hist(data, bins=20, color='purple', edgecolor='white')
plt.title("Histogram", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# 6. Pie Chart
plt.figure(6)
labels = ['Python', 'Java', 'C++', 'Ruby']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)
plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=140,
        textprops={'fontsize': 12})
plt.title("Pie Chart", fontsize=14)
plt.axis('equal')
plt.show()

# 7. Box Plot
plt.figure(7)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data, patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='black'),
           medianprops=dict(color='red'))
plt.title("Box Plot", fontsize=14)
plt.xlabel("Data Sets", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.show()

# 8. Multiple Line Plots
plt.figure(8)
x = [1, 2, 3, 4]
y1 = [10, 20, 25, 30]
y2 = [5, 15, 20, 25]
plt.plot(x, y1, label='Line 1', color='blue', marker='o')
plt.plot(x, y2, label='Line 2', color='green', marker='s')
plt.title("Multiple Line Plots", fontsize=14)
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 9. Subplots
plt.figure(9)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
axs[0, 0].plot([1, 2], [1, 2], color='blue')
axs[0, 0].set_title("Line Plot", fontsize=12)

# Plot 2
axs[0, 1].bar([1, 2], [3, 4], color='orange')
axs[0, 1].set_title("Bar Chart", fontsize=12)

# Plot 3
axs[1, 0].hist(np.random.randn(100), color='green', bins=15)
axs[1, 0].set_title("Histogram", fontsize=12)

# Plot 4
axs[1, 1].scatter([1, 2], [2, 1], color='red', s=100)
axs[1, 1].set_title("Scatter Plot", fontsize=12)

plt.suptitle("Subplot Example", fontsize=16)
plt.tight_layout()
plt.show()
"""
    return code


def logistic_regression():
    """
    Returns complete Logistic Regression implementation code as a string.
    Includes data loading, model training, evaluation, and visualization.
    Usage:
    code = logistic_regression()
    print(code)  # To view the code
    exec(code)   # To execute the code
    """
    code = """
# Logistic Regression Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error
)

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn')

# 1. Load and prepare data
print("\\n=== Loading Dataset ===")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(f"\\nDataset shape: {df.shape}")
print("\\nFirst 5 rows:")
print(df.head())

# 2. Feature selection and train-test split
print("\\n=== Preparing Data ===")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# 3. Model training
print("\\n=== Training Model ===")
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)
print("\\nModel trained successfully!")

# 4. Model evaluation
print("\\n=== Model Evaluation ===")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\\nAccuracy: {accuracy:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("\\nConfusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# 5. Visualization
print("\\n=== Generating Visualizations ===")
plt.figure(figsize=(12, 6))

# Actual vs Predicted plot
plt.subplot(1, 2, 1)
plt.plot(y_test.values, label='Actual', marker='o', linestyle='', alpha=0.7)
plt.plot(y_pred, label='Predicted', marker='x', linestyle='', alpha=0.7)
plt.title("Actual vs Predicted Values", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Target Class", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Confusion Matrix plot
plt.subplot(1, 2, 2)
conf_mat = confusion_matrix(y_test, y_pred)
plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.yticks([0, 1], ['Benign', 'Malignant'])

plt.tight_layout()
plt.show()

# Return model if needed
# return model, y_pred, y_prob
"""
    return code

def auc():
    """
    Returns Logistic Regression with ROC analysis code as provided.
    Usage:
    code = logistic_roc()
    print(code)  # To view the code
    exec(code)   # To execute the code
    """
    code = """
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1

# Calculate AUC score
auc = roc_auc_score(y_test, y_prob)
print(f"AUC Score: {auc:.2f}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
"""
    return code