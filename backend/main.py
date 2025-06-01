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
from sklearn.neighbors import NearestNeighbors  # <<< ENHANCEMENT
import scipy.sparse as sp
import numpy as np
import pickle  # <<< ENHANCEMENT: for saving/loading the vectorizer
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping 
from numpy.linalg import norm

# Download required NLTK data
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

df = pd.DataFrame(data)

if "programme" not in df.columns:
    raise Exception("Expected 'programme' field in JSON data.")

# 1) Clean + Tokenize
df["cleaned_programme"] = df["programme"].apply(clean_text)
df["tokens"] = df["cleaned_programme"].apply(tokenize_text)

# 2) TF-IDF Vectorizer: try to load from disk, otherwise fit and save
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    vectorizer.fit(df["cleaned_programme"])
    os.makedirs("models", exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

tfidf_matrix = vectorizer.transform(df["cleaned_programme"])

# 3) Optional One-Hot Encoding of categorical columns
if {"interest_area", "university"}.issubset(df.columns):
    categorical_columns = ['interest_area', 'university']
    encoder = OneHotEncoder(sparse_output=False)
    categorical_data = df[categorical_columns]
    encoded_features = encoder.fit_transform(categorical_data)
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    df = pd.concat([df, encoded_df], axis=1)

    encoded_sparse = sp.csr_matrix(encoded_df.values)
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

# 1) Load or train the encoder
encoder_model_path = os.path.join("models", "encoder_model.h5")
if os.path.exists(encoder_model_path):
    print("Loading existing encoder model...")
    encoder = load_model(encoder_model_path)
else:
    print("Training new model...")
    encoder = train_model(combined_features)

# ---------- Precompute Programme Embeddings + Build ANN Index ----------

# 1) Compute 32-dim embeddings for each programme
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

programme_embeddings = np.vstack(programme_embeddings)  # shape = (num_programmes, 32)

# 2) Build a NearestNeighbors index on cosine distance
#    Note: NearestNeighbors with metric='cosine' actually returns distance = 1 - cosine_similarity.
nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
nn_model.fit(programme_embeddings)

# ---------- Utility Functions ----------

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b))

def process_user_payload(input_data: dict):
    """
    Processes the extended user payload into a TF-IDF vector.
    Currently combines 'preferred_programme' and 'additional_info' texts.
    """
    text_fields = []
    if "preferred_programme" in input_data:
        text_fields.append(input_data["preferred_programme"])
    if "additional_info" in input_data:
        text_fields.append(input_data["additional_info"])
    
    combined_text = " ".join(text_fields) if text_fields else ""
    cleaned_text = clean_text(combined_text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    return text_vector

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
    
    tfidf_vector = vectorizer.transform([cleaned]).toarray()
    # Pad to match training dim if needed
    if tfidf_vector.shape[1] < expected_dim:
        missing = expected_dim - tfidf_vector.shape[1]
        tfidf_vector = np.hstack([tfidf_vector, np.zeros((1, missing))])
    
    latent_embedding = encoder.predict(tfidf_vector)
    return {"latent_embedding": latent_embedding.tolist()}

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
    # 1) Turn user payload into a TF-IDF vector
    raw_user_vector = process_user_payload(input_data)
    if raw_user_vector.shape[1] < expected_dim:
        missing = expected_dim - raw_user_vector.shape[1]
        raw_user_vector = np.hstack([raw_user_vector, np.zeros((1, missing))])
    
    # 2) Encode into 32-dim latent space
    user_embedding = encoder.predict(raw_user_vector)[0]
    
    # 3) Query the NearestNeighbors index for top_n nearest programmes
    #    distances[i] = 1 - cosine_similarity(user_embedding, programme_embeddings[index[i]])
    distances, indices = nn_model.kneighbors([user_embedding], n_neighbors=top_n)
    recommended_indices = indices[0].tolist()
    recommendations = df.iloc[recommended_indices].to_dict(orient="records")
    
    return {"recommendations": recommendations}

@app.post("/train")
def retrain_model():
    """
    Retrains the autoencoder model on the current dataset.
    WARNING: This will overwrite the existing model files.
    """
    global encoder, combined_features, programme_embeddings, nn_model
    encoder = train_model(combined_features)

    # Re-compute programme embeddings
    programme_embeddings = []
    for i in range(len(df)):
        text = df.iloc[i]["cleaned_programme"]
        tfidf_vector = vectorizer.transform([text]).toarray()
        if tfidf_vector.shape[1] < expected_dim:
            missing = expected_dim - tfidf_vector.shape[1]
            tfidf_vector = np.hstack([tfidf_vector, np.zeros((1, missing))])
        latent_embedding = encoder.predict(tfidf_vector)
        programme_embeddings.append(latent_embedding[0])
    programme_embeddings = np.vstack(programme_embeddings)

    # Rebuild the NearestNeighbors index
    nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn_model.fit(programme_embeddings)

    return {"detail": "Model retrained successfully."}

# To run the server:
# python -m uvicorn main:app --reload
