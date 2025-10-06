from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = load_model("lstm_model.h5", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Example: categorical columns from your CSV
categorical_columns = [
    'side', 'position', 'player', 'team', 'champion',
    'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
]

scaler = StandardScaler()

@app.get("/")
async def root():
    return {"message": "League Match Predictor API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            return JSONResponse({"error": "Please upload a CSV file"}, status_code=400)

        # Load uploaded CSV
        df = pd.read_csv(file.file)

        # Keep a copy of original names for response
        df_original = df.copy()

        # Encode categorical values for model
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Convert other columns to numeric
        for col in df.columns:
            if col not in categorical_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Scale features
        features_scaled = scaler.fit_transform(df)

        # Match model input shape
        expected_features = model.input_shape[2]
        current_features = features_scaled.shape[1]
        if current_features > expected_features:
            features_scaled = features_scaled[:, :expected_features]
        elif current_features < expected_features:
            features_scaled = np.pad(
                features_scaled,
                ((0, 0), (0, expected_features - current_features)),
                "constant"
            )

        # Reshape for LSTM input
        X = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

        # Predict
        preds = model.predict(X).flatten()
        results = (preds > 0.5).astype(int).tolist()

        # Attach predictions to original dataframe
        df_original['predicted_result'] = results
        df_original['win_probability'] = (preds * 100).round(2)

        # Team-level summary
        team_probs = df_original.groupby('team')['win_probability'].mean().to_dict()

        # Build response (using human-readable names)
        response = {
            "team_probabilities": team_probs,
            "players": df_original[['side', 'position', 'player', 'team', 'champion',
                                    'win_probability', 'predicted_result']].to_dict(orient='records')
        }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
