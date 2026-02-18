"""
Ingredient Analyzer Backend
Flask API for predicting ingredient safety using TensorFlow/Keras model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained Keras model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ingredient_model.keras')

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def get_risk_info(prediction):
    """
    Convert model prediction to risk score and label
    
    Args:
        prediction: Model output (assumed to be a probability or class index)
    
    Returns:
        dict with risk_score and risk_label
    """
    # Assuming prediction is a single value between 0 and 1
    # or a probability array. Adjust based on your model output.
    
    if isinstance(prediction, np.ndarray):
        pred_value = float(prediction[0])
    else:
        pred_value = float(prediction)
    
    # Determine risk label based on prediction value
    # Adjust thresholds based on your model's output range
    if pred_value < 0.33:
        risk_label = "Safe"
        risk_score = round(pred_value * 100, 2)
    elif pred_value < 0.66:
        risk_label = "Moderate Risk"
        risk_score = round(pred_value * 100, 2)
    else:
        risk_label = "Harmful"
        risk_score = round(pred_value * 100, 2)
    
    return {
        "risk_score": risk_score,
        "risk_label": risk_label
    }


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON input:
    {
        "features": [feature1, feature2, ...]
    }
    
    OR for ingredient analysis:
    {
        "ingredients": ["ingredient1", "ingredient2", ...]
    }
    
    Returns:
    {
        "prediction": ...,
        "risk_score": ...,
        "risk_label": "Safe" | "Moderate Risk" | "Harmful"
    }
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please ensure ingredient_model.keras exists."
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Extract features based on input format
        # Option 1: Direct features array
        if "features" in data:
            features = np.array(data["features"])
        # Option 2: Ingredients list (would need preprocessing)
        elif "ingredients" in data:
            # For ingredient analysis, convert to numerical features
            # This is a placeholder - adjust based on your model's expected input
            ingredients = data["ingredients"]
            features = np.array([hash(ing) % 1000 for ing in ingredients])
            features = features.reshape(1, -1)
        else:
            return jsonify({
                "error": "Invalid input format. Expected 'features' or 'ingredients' key."
            }), 400
        
        # Reshape if needed (model expects specific input shape)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get risk information
        risk_info = get_risk_info(prediction)
        
        # Return JSON response
        return jsonify({
            "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else float(prediction),
            "risk_score": risk_info["risk_score"],
            "risk_label": risk_info["risk_label"]
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Ingredient Analyzer API",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Health check"
        }
    })


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
