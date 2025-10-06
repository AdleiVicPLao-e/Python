from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model and preprocessing objects
try:
    model = load_model("lstm_model.h5", compile=False)
    print("✅ Model loaded successfully")
    
    # Load preprocessing objects
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl') 
    label_encoders = joblib.load('label_encoders.pkl')
    print("✅ Preprocessing objects loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model or preprocessing objects: {e}")
    model = None
    scaler = None
    pca = None
    label_encoders = None

# Define categorical columns (must match your training)
categorical_columns = ['side', 'position', 'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5']

# Define all columns used in training (from your training code)
columns_to_use = [
    'side', 'position', 'player', 'team', 'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'gamelength',
    'k', 'd', 'a', 'teamkills', 'teamdeaths', 'doubles', 'triples', 'quadras', 'pentas',
    'fb', 'fbassist', 'fbvictim', 'fbtime', 'kpm', 'okpm', 'ckpm', 'fd', 'fdtime',
    'teamdragkills', 'oppdragkills', 'elementals', 'oppelementals', 'firedrakes', 'waterdrakes',
    'earthdrakes', 'airdrakes', 'elders', 'oppelders', 'herald', 'heraldtime', 'ft', 'fttime',
    'firstmidouter', 'firsttothreetowers', 'teamtowerkills', 'opptowerkills', 'fbaron', 'fbarontime',
    'teambaronkills', 'oppbaronkills', 'dmgtochamps', 'dmgtochampsperminute', 'dmgshare',
    'earnedgoldshare', 'wards', 'wpm', 'wardshare', 'wardkills', 'wcpm', 'visionwards',
    'visionwardbuys', 'visiblewardclearrate', 'invisiblewardclearrate', 'totalgold', 'earnedgpm',
    'goldspent', 'gspd', 'minionkills', 'monsterkills', 'monsterkillsownjungle',
    'monsterkillsenemyjungle', 'cspm', 'goldat10', 'oppgoldat10', 'gdat10', 'goldat15', 'oppgoldat15',
    'gdat15', 'xpat10', 'oppxpat10', 'xpdat10', 'csat10', 'oppcsat10', 'csdat10', 'csat15', 'oppcsat15', 'csdat15'
]

@app.get("/")
async def root():
    return {
        "message": "League Match Predictor API", 
        "status": "active", 
        "model_loaded": model is not None,
        "preprocessing_loaded": scaler is not None and pca is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "preprocessing_loaded": scaler is not None and pca is not None
    }

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if model is None or scaler is None or pca is None:
            return JSONResponse({
                "error": "Model or preprocessing objects not loaded properly"
            }, status_code=500)
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            return JSONResponse({"error": "Please upload a CSV file"}, status_code=400)

        # Load uploaded CSV
        df = pd.read_csv(file.file)
        df_original = df.copy()

        # Ensure all required columns are present
        missing_columns = [col for col in columns_to_use if col not in df.columns]
        if missing_columns:
            return JSONResponse({
                "error": f"Missing required columns: {missing_columns}"
            }, status_code=400)

        # Select only the columns used in training
        df = df[columns_to_use]

        # Encode categorical values using saved label encoders
        for col in categorical_columns:
            if col in df.columns:
                if col in label_encoders:
                    # Transform using saved encoder
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                else:
                    # Fallback: create new encoder if not found
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))

        # Convert other columns to numeric
        for col in df.columns:
            if col not in categorical_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Scale features using saved scaler
        features_scaled = scaler.transform(df)

        # Apply PCA using saved PCA
        features_pca = pca.transform(features_scaled)

        # Reshape for LSTM input [samples, time steps, features]
        X = np.reshape(features_pca, (features_pca.shape[0], 1, features_pca.shape[1]))

        # Predict
        preds = model.predict(X).flatten()
        results = (preds > 0.5).astype(int).tolist()

        # Attach predictions to original dataframe
        df_original['predicted_result'] = results
        df_original['win_probability'] = (preds * 100).round(2)

        # Team-level summary
        team_probs = df_original.groupby('team')['win_probability'].mean().to_dict()

        # Build response
        response = {
            "team_probabilities": team_probs,
            "players": df_original[['side', 'position', 'player', 'team', 'champion',
                                    'win_probability', 'predicted_result']].to_dict(orient='records')
        }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Demo endpoint for testing without CSV
@app.post("/predict_demo")
async def predict_demo():
    """Demo endpoint that uses sample data for testing"""
    try:
        if model is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=500)
        
        # Create sample data matching your training format
        sample_data = {
            'side': ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Red', 'Red', 'Red', 'Red', 'Red'],
            'position': ['Top', 'Jungle', 'Middle', 'ADC', 'Support', 'Top', 'Jungle', 'Middle', 'ADC', 'Support'],
            'player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
            'team': ['Team A', 'Team A', 'Team A', 'Team A', 'Team A', 'Team B', 'Team B', 'Team B', 'Team B', 'Team B'],
            'champion': ['Aatrox', 'LeeSin', 'Orianna', 'Ezreal', 'Leona', 'Gnar', 'Sejuani', 'Syndra', 'Kaisa', 'Nautilus'],
            'ban1': ['Yasuo'] * 10,
            'ban2': ['Zed'] * 10,
            'ban3': ['Akali'] * 10,
            'ban4': ['Irelia'] * 10,
            'ban5': ['Fiora'] * 10,
            'k': [2, 1, 3, 4, 1, 1, 2, 2, 3, 0],
            'd': [1, 2, 1, 0, 3, 2, 1, 3, 1, 2],
            'a': [4, 5, 2, 3, 6, 1, 4, 2, 3, 5]
        }
        
        # Fill missing columns with zeros
        for col in columns_to_use:
            if col not in sample_data:
                sample_data[col] = [0] * 10
        
        df = pd.DataFrame(sample_data)
        return await predict_csv_internal(df)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)