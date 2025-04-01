from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("demand_model.pkl")
category_encoder = joblib.load("category_encoder.pkl")
brand_encoder = joblib.load("brand_encoder.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure all required fields exist
        required_fields = ["initial_price", "final_price", "rating", "category", "brand"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract and convert values
        initial_price = float(data["initial_price"])
        final_price = float(data["final_price"])
        rating = float(data["rating"])
        category = data["category"]
        brand = data["brand"]

        # Compute discount percentage
        discount_percentage = ((initial_price - final_price) / initial_price) * 100 if initial_price > 0 else 0

        # Adjust rating
        adjusted_rating = rating + 1  # Prevent zero ratings

        # Encode category and brand (handle unknowns safely)
        category_encoded = category_encoder.transform([category])[0] if category in category_encoder.classes_ else -1
        brand_encoded = brand_encoder.transform([brand])[0] if brand in brand_encoder.classes_ else -1

        # Ensure valid encoding (prevent crashes)
        if category_encoded == -1 or brand_encoded == -1:
            return jsonify({"error": "Invalid category or brand"}), 400

        # Create feature array
        features = np.array([[final_price, discount_percentage, adjusted_rating, 0, category_encoded, brand_encoded]])

        # Predict demand score
        demand_score = model.predict(features)[0]

        return jsonify({"demand_score": round(demand_score, 4)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
