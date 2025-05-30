# main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping 
from numpy.linalg import norm



# Download required NLTK data (including 'punkt' and 'punkt_tab')
nltk.download('punkt')
nltk.download('punkt_tab')

app = FastAPI(title="EduPath Programme Recommender API")

# ---------- Data Processing Functions ----------

def clean_text(text: str) -> str:
    """
    Lowercase the text, remove punctuation and extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str):
    """
    Tokenizes the cleaned text into words.
    """
    return word_tokenize(text)

# ---------- Load and Process Data ----------

DATA_PATH = os.path.join("data", "programmes.json")
if not os.path.exists(DATA_PATH):
    raise Exception(f"Data file not found at {DATA_PATH}")

with open(DATA_PATH, "r") as file:
    data = json.load(file)

# Assume JSON data is a list of dictionaries.
df = pd.DataFrame(data)

# Use the "programme" field as the text description.
if "programme" not in df.columns:
    raise Exception("Expected 'programme' field in JSON data.")

df["cleaned_programme"] = df["programme"].apply(clean_text)
df["tokens"] = df["cleaned_programme"].apply(tokenize_text)

# Create TF-IDF features from the cleaned programme text.
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
tfidf_matrix = vectorizer.fit_transform(df["cleaned_programme"])

# Optionally, encode additional categorical fields (e.g., interest_area, university)
if {"interest_area", "university"}.issubset(df.columns):
    categorical_columns = ['interest_area', 'university']
    encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
    categorical_data = df[categorical_columns]
    encoded_features = encoder.fit_transform(categorical_data)
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    # Merge encoded categorical features into the dataframe
    df = pd.concat([df, encoded_df], axis=1)
    # Convert one-hot encoded DataFrame to sparse matrix
    encoded_sparse = sp.csr_matrix(encoded_df.values)
    # Combine TF-IDF features with categorical features into a dense array
    combined_features = sp.hstack([tfidf_matrix, encoded_sparse]).toarray()
else:
    combined_features = tfidf_matrix.toarray()

# ---------- Model Training and Inference ----------

def build_autoencoder(input_dim, encoding_dim=32):
    """
    Builds a simple autoencoder with one hidden (encoder) layer and one output layer.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder_model = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder_model

def train_model(X):
    """
    Trains the autoencoder model on the input features and saves the trained models.
    """
    input_dim = X.shape[1]
    autoencoder, encoder_model = build_autoencoder(input_dim, encoding_dim=32)
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    autoencoder.fit(X, X, epochs=50, batch_size=8, shuffle=True,
                    callbacks=[early_stopping], verbose=1)
    
    os.makedirs("models", exist_ok=True)
    autoencoder.save(os.path.join("models", "autoencoder_model.h5"))
    encoder_model.save(os.path.join("models", "encoder_model.h5"))
    return encoder_model

# If a trained encoder model exists, load it; otherwise, train a new one.
encoder_model_path = os.path.join("models", "encoder_model.h5")
if os.path.exists(encoder_model_path):
    print("Loading existing encoder model...")
    encoder = load_model(encoder_model_path)
else:
    print("Training new model...")
    encoder = train_model(combined_features)

# ---------- API Endpoints ----------

@app.get("/programmes", response_model=list)
def get_all_programmes():
    """
    Returns all programme records.
    """
    return df.to_dict(orient="records")

@app.get("/programmes/{programme_id}")
def get_programme(programme_id: int):
    """
    Returns the details of a specific programme by index.
    """
    if programme_id < 0 or programme_id >= len(df):
        raise HTTPException(status_code=404, detail="Programme not found")
    return df.iloc[programme_id].to_dict()

@app.post("/predict")
def predict_programme(input_data: dict):
    """
    Receives a JSON payload with a 'programme' field,
    preprocesses it, and returns the latent embedding from the encoder.
    Example payload:
    {
        "programme": "Bachelor of Education in English"
    }
    """
    if "programme" not in input_data:
        raise HTTPException(status_code=400, detail="Missing 'programme' field.")
    
    raw_text = input_data["programme"]
    cleaned = clean_text(raw_text)
    
    # Transform the input text using the TF-IDF vectorizer.
    tfidf_vector = vectorizer.transform([cleaned]).toarray()
    
    # If categorical fields were used during training, ensure dimensions match.
    expected_dim = combined_features.shape[1]
    if tfidf_vector.shape[1] < expected_dim:
        missing = expected_dim - tfidf_vector.shape[1]
        tfidf_vector = np.hstack([tfidf_vector, np.zeros((1, missing))])
    
    # Get the latent embedding using the encoder model.
    latent_embedding = encoder.predict(tfidf_vector)
    return {"latent_embedding": latent_embedding.tolist()}
  
  



def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b))

def process_user_payload(input_data: dict):
    """
    Processes the extended user payload into a TF-IDF vector.
    For now, we only combine the text from 'preferred_programme' and 'additional_info'.
    Extend this function to include user_core_subjects, user_elective_subjects, and user_grades.
    """
    text_fields = []
    if "preferred_programme" in input_data:
        text_fields.append(input_data["preferred_programme"])
    if "additional_info" in input_data:
        text_fields.append(input_data["additional_info"])
    
    combined_text = " ".join(text_fields) if text_fields else ""
    cleaned_text = clean_text(combined_text)
    # Generate the TF-IDF vector
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    return text_vector

# Precompute latent embeddings for all programmes at startup
programme_embeddings = []
expected_dim = combined_features.shape[1]
for i in range(len(df)):
    text = df.iloc[i]["cleaned_programme"]
    tfidf_vector = vectorizer.transform([text]).toarray()
    if tfidf_vector.shape[1] < expected_dim:
        missing = expected_dim - tfidf_vector.shape[1]
        tfidf_vector = np.hstack([tfidf_vector, np.zeros((1, missing))])
    latent_embedding = encoder.predict(tfidf_vector)
    programme_embeddings.append(latent_embedding[0])

@app.post("/recommend")
def recommend_programmes(input_data: dict, top_n: int = 5):
    """
    Extended recommendation endpoint that uses user profile information.
    Example payload:
    {
        "preferred_programme": "Engineering",
        "user_core_subjects": ["Mathematics", "Physics", "English"],
        "user_elective_subjects": ["Chemistry", "Biology"],
        "user_grades": {"Mathematics": "A1", "Physics": "B2", "English": "A1", "Chemistry": "B3", "Biology": "C6"},
        "additional_info": "I prefer hands-on learning and practical projects."
    }
    """
    # Process the user payload to get a TF-IDF vector
    raw_user_vector = process_user_payload(input_data)
    
    # Ensure the vector is the same dimension as training data
    if raw_user_vector.shape[1] < expected_dim:
        missing = expected_dim - raw_user_vector.shape[1]
        raw_user_vector = np.hstack([raw_user_vector, np.zeros((1, missing))])
    
    # Pass the TF-IDF vector through the encoder to get the latent embedding (32-d)
    user_embedding = encoder.predict(raw_user_vector)[0]
    
    # Compute cosine similarity between user_embedding and each programme's latent embedding
    similarities = []
    for idx, prog_embed in enumerate(programme_embeddings):
        sim = cosine_similarity(user_embedding, prog_embed)
        similarities.append((idx, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    recommended_indices = [idx for idx, sim in similarities[:top_n]]
    recommendations = df.iloc[recommended_indices].to_dict(orient="records")
    
    return {"recommendations": recommendations}












@app.post("/train")
def retrain_model():
    """
    Retrains the autoencoder model on the current dataset.
    WARNING: This will overwrite the existing model files.
    """
    global encoder, combined_features
    encoder = train_model(combined_features)
    return {"detail": "Model retrained successfully."}

# To run the server:
# python -m uvicorn main:app --reload
