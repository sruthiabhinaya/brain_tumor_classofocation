import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

dataset_dir = r"D:\brh35\BrainTumor17Class"

# Load fused features
X_train_fused = np.load(f"{dataset_dir}/X_train_fused.npy")
X_test_fused = np.load(f"{dataset_dir}/X_test_fused.npy")
y_train = np.load(f"{dataset_dir}/y_train_balanced.npy")
y_test = np.load(f"{dataset_dir}/y_test.npy")

# ----------------------------
# Feature selection
# ----------------------------
selector = SelectKBest(f_classif, k=2000)
X_train_selected = selector.fit_transform(X_train_fused, y_train)
X_test_selected = selector.transform(X_test_fused)

# Save selector
joblib.dump(selector, f"{dataset_dir}/feature_selector.pkl")
print("Feature selector saved!")

# ----------------------------
# Train classifiers
# ----------------------------
classifiers = {
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    results[name] = (y_pred, acc)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(xticks_rotation='vertical')
    plt.title(f"{name} Confusion Matrix")
    plt.show()
    
    # Classification report
    print(classification_report(y_test, y_pred))

# Save best model (XGBoost)
best_model = classifiers['XGBoost']
joblib.dump(best_model, f"{dataset_dir}/xgboost_classifier.pkl")
print("XGBoost model saved!")
