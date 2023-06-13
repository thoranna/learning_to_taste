import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms.functional as F
from torchvision import transforms
import PIL
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import requests
import nltk
import pandas as pd
from io import BytesIO
import random
import torch
import torch.nn as nn
import h5py
from transformers import set_seed

def set_deterministic(seed=42):
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # This is to force PyTorch to use deterministic algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Transformers
    set_seed(seed)

# You might need to download the stopwords first
nltk.download('punkt')
nltk.download('stopwords')

# Default to cpu for now
device = 'cpu'
MAX_IMG_SAMPLES = 100

def fit_model(model_name, data):
    set_deterministic(seed=42)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)

    data["review"] = data["review"].apply(preprocess_text)
    reviews = data["review"].values
    image_files = data["image"].apply(lambda x: x.replace('/p/', '') if pd.notnull(x) else x).values

    experiment_ids = data["experiment_id"].values
    unique_experiment_ids = data["experiment_id"].unique()

    with h5py.File('embeddings.hdf5', 'a') as hf:
        if "mean_embeddings" in hf.keys():
            del hf["mean_embeddings"]
        for i, experiment_id in enumerate(unique_experiment_ids):
            if "embeddings" in hf.keys():
                del hf["embeddings"]
            print("i: ", i)
            idx = (experiment_ids == experiment_id)
            vintage_reviews = reviews[idx]
            vintage_image_files = image_files[idx]
            # For images
            try:
                vintage_image_files = random.sample(list(vintage_image_files), MAX_IMG_SAMPLES)
            except: 
                vintage_image_files = list(vintage_image_files)
            # Use your existing function for image processing
            download_process_delete_images(
                model=model,
                urls=vintage_image_files,
                reviews=vintage_reviews ,
                processor=processor,
                device=device)

            # Load embeddings from the file
            embeddings = hf['embeddings'][:]  # This is a reference to the data on disk
            vintage_embedding = embeddings.mean(axis=0)
            print("vintage embed: ", vintage_embedding)
            # Combine embeddings
            try:
                hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1), axis=0)
                hf["mean_embeddings"][-1:] = vintage_embedding
            except KeyError:
                # If 'mean_embeddings' dataset doesn't exist, create it
                hf.create_dataset("mean_embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))
    
        # At the end, load all embeddings at once as a memmap array
        embedding_matrix = hf["mean_embeddings"][:]
        return embedding_matrix, unique_experiment_ids


def download_process_delete_images(model, urls, reviews, processor, device, batch_size=10, embedding_dim=512, embedding_path='embeddings.hdf5'):
    with h5py.File(embedding_path, 'a') as hf:    
        if "embeddings" not in hf:
            hf.create_dataset("embeddings", data=np.empty((0, embedding_dim)), maxshape=(None, embedding_dim))
        for batch_start in range(0, len(urls), batch_size):
            batch_urls = urls[batch_start: batch_start + batch_size]
            batch_reviews = reviews[batch_start: batch_start + batch_size]
            for i, url in enumerate(batch_urls):
                if pd.isnull(url):
                    emb = torch.zeros((1, embedding_dim)).to(device)
                else:
                    url = "https://images.vivino.com/labels/" + url
                    url = url.replace('p/', '')
                    response = requests.get(url)
                    try:
                        img = Image.open(BytesIO(response.content))
                        review = batch_reviews[i]
                        max_length = 77
                        truncated_review = review[:max_length]
                        inputs = processor(text=[truncated_review], images=img, return_tensors="pt", padding=True)
                        outputs = model(**inputs)
                        emb = outputs.text_embeds + outputs.image_embeds
                    except Exception as e:  # Catch all exceptions, not just UnidentifiedImageError
                        print("in the exception")
                        print(e)
                        emb = torch.zeros((1, embedding_dim)).to(device)

                vintage_embedding = emb.detach().cpu().numpy()
                try:
                    hf["embeddings"].resize((hf["embeddings"].shape[0] + 1), axis=0)
                    hf["embeddings"][-1:] = vintage_embedding
                except KeyError:
                    hf.create_dataset("embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))


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