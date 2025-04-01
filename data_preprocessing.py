import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("cleaned_shein_products.csv")  # Ensure correct file

# Show all column names
print("Columns in dataset:", df.columns.tolist())

# Selecting relevant features
df = df[['product_name', 'brand', 'category', 'initial_price', 'final_price', 'rating']]

# Drop duplicates
df = df.drop_duplicates()

# Encode categorical variables (Brand & Category)
le_brand = LabelEncoder()
le_category = LabelEncoder()

df["brand_encoded"] = le_brand.fit_transform(df["brand"])
df["category_encoded"] = le_category.fit_transform(df["category"])

# Save label encoders for later use
with open("label_encoders.pkl", "wb") as file:
    pickle.dump({"brand": le_brand, "category": le_category}, file)

# Save cleaned dataset
df.to_csv("cleaned_shein_products.csv", index=False)

print("✅ Data Preprocessing Done. Cleaned data saved!")
