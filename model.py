# ================================
# IMPORTING LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ================================
# LOAD DATA
# ================================
df = pd.read_csv('/content/heart.csv')

# ================================
# DATA CLEANING
# ================================
df_clean = df.copy()
df_clean = df_clean[df_clean['RestingBP'] != 0]

no_hd = df_clean['HeartDisease'] == 0
df_clean.loc[no_hd, 'Cholesterol'] = df_clean.loc[no_hd, 'Cholesterol'].replace(0, df_clean.loc[no_hd, 'Cholesterol'].mean())
df_clean.loc[~no_hd, 'Cholesterol'] = df_clean.loc[~no_hd, 'Cholesterol'].replace(0, df_clean.loc[~no_hd, 'Cholesterol'].mean())

df_clean = pd.get_dummies(df_clean, drop_first=True)


# ================================
# DATA SPLITTING
# ================================
X = df_clean.drop('HeartDisease', axis=1)
y = df_clean['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ================================
# MODEL TRAINING & EVALUATION
# ================================

# --- Random Forest ---
rf = RandomForestClassifier(class_weight='balanced', n_estimators=200)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {round(acc_rf * 100, 2)}%")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
joblib.dump(rf, "heart_disease_rf_model.pkl")
print("Random Forest model saved")

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {round(acc_lr * 100, 2)}%")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens')
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Support Vector Machine (SVM) ---
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {round(acc_svm * 100, 2)}%")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Oranges')
plt.title('SVM - Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- MLP Classifier ---
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.001, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Classifier Accuracy: {round(acc_mlp * 100, 2)}%")
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))
sns.heatmap(confusion_matrix(y_test, y_pred_mlp), annot=True, fmt='d', cmap='Purples')
plt.title('MLP Classifier - Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
