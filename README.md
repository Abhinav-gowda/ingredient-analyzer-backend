# Ingredient Analyzer Backend

## IMPORTANT: Add Your Trained Model

You need to place your trained Keras model file in this folder.

### Required File:
- **filename:** `ingredient_model.keras`
- **description:** Your trained TensorFlow/Keras model file

### How to add your model:
1. Copy your trained model file (e.g., from your training notebook/output)
2. Rename it to `ingredient_model.keras`
3. Place it in this folder (ingredient-analyzer-backend/)

The app.py file is configured to load this model automatically when it starts.

## Running the Backend

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Flask app:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions

### Example Prediction Request:
```json
{
    "features": [0.1, 0.2, 0.3, ...]
}
```

Or for ingredient analysis:
```json
{
    "ingredients": ["sugar", "salt", "preservatives"]
}
```

## Deployment

For production deployment with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```
