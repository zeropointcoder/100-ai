import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))

# Feature importance
feat_importance = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:", feat_importance)

# Save model
joblib.dump(model, "iris_model.joblib")
print("\nModel saved as iris_model.joblib\n")