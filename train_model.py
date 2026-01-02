import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset (use relative path for deployment)
df = pd.read_csv("data/dataset.csv")

# Separate features & target
X = df.drop(columns=["fruit_name"])
y = df["fruit_name"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# Save model + feature columns
with open("model/random_forest_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("✅ Random Forest model trained & saved successfully")
