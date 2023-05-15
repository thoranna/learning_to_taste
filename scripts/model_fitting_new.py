import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from transformers import T5EncoderModel
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import open_clip

# You might need to download the stopwords first
nltk.download('punkt')
nltk.download('stopwords')

def fit_model(model_to_fit, data):
    global model_fitting
    if model_to_fit == 'clip':
        model_name = 'ViT-B-32'
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif model_to_fit == 't5_small':
        model_name = 't5-small'
        model = T5EncoderModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_to_fit == 'distil_bert':
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Invalid model_to_fit value. Choose from ['distil_bert', 'albert_base', 'mobilebert', 'distilroberta', 'clip']")

    # preprocessing step
    data["review"] = data["review"].apply(preprocess_text)
    experiment_ids = data["experiment_id"].values
    reviews = data["review"].values
    region_ids = data["region_id"].values

    unique_experiment_ids = data["experiment_id"].unique()
    vintage_embeddings = {}
    vintage_region_ids = {}

    for experiment_id in unique_experiment_ids:
        idx = (experiment_ids == experiment_id)
        vintage_reviews = reviews[idx]
        vintage_region_id = region_ids[idx][0]
        vintage_embedding = create_vintage_embedding(tokenizer, model, [experiment_id] * len(vintage_reviews), vintage_reviews, model_to_fit).mean(axis=0)

        vintage_embeddings[experiment_id] = vintage_embedding
        vintage_region_ids[experiment_id] = vintage_region_id

    print(list(vintage_embeddings.values()))
    embedding_matrix = np.array(list(vintage_embeddings.values()), dtype=object)
    return embedding_matrix, unique_experiment_ids

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if not word in stop_words]
    
    # Join words back into string
    text = ' '.join(text)
    
    return text

def create_vintage_embedding(tokenizer, model, vintage_ids, reviews, model_name, batch_size=10):
    embeddings = []
    for i in range(0, len(vintage_ids), batch_size):
        batch_vintage_ids = vintage_ids[i:i+batch_size]
        batch_reviews = reviews[i:i+batch_size]
        input_text = [f"vintage {vintage_id}: {review}" for vintage_id, review in zip(batch_vintage_ids, batch_reviews)]

        if model_name is not 'clip':
            input_tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = model(**input_tokenized)
            batch_embeddings = output.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(batch_embeddings)
        else:
            input_tokenized = tokenizer(input_text)
            output = model.encode_text(input_tokenized)
            embeddings.append(output)
    return np.vstack(embeddings)