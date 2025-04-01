import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("cleaned_shein_products.csv")

# Compute discount percentage (avoid division errors)
df["discount_percentage"] = np.where(
    df["initial_price"] > 0,
    ((df["initial_price"] - df["final_price"]) / df["initial_price"]) * 100,
    0
)

# Normalize rating
df["adjusted_rating"] = df["rating"] + 1  # Prevent zero ratings

# Compute category popularity
df["category_popularity"] = df["category"].map(df["category"].value_counts(normalize=True))

# Create a demand score (ensure it doesn’t go negative)
df["demand_score"] = np.maximum(
    (df["discount_percentage"] * 0.4) + 
    (df["adjusted_rating"] * 0.3) + 
    (df["category_popularity"] * 0.3), 
    0
)

# Features and Target
features = ["final_price", "discount_percentage", "adjusted_rating", "category_popularity", "category_encoded", "brand_encoded"]
target = "demand_score"
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Save the model
joblib.dump(model, "demand_model.pkl")

print("✅ Model training complete. Model saved!")
