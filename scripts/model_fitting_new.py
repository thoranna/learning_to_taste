import torch
import numpy as np
from transformers import T5EncoderModel, AutoTokenizer, AlbertTokenizer, AlbertModel, \
    BartTokenizer, BartModel, PegasusTokenizer, PegasusForConditionalGeneration, \
    AutoModel, CLIPModel, ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, \
    DeiTForImageClassificationWithTeacher
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, CLIPTextModel
from transformers import AutoProcessor, CLIPVisionModel
from torchvision import models
import torchvision.transforms.functional as F
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
import os

# You might need to download the stopwords first
nltk.download('punkt')
nltk.download('stopwords')

text_models = ['t5_small', 'albert', 'bart', 'pegasus', 'distil_bert', 'clip_text']
image_models = ['vit_base', 'deit_small', 'resnet', 'clip_image']
image_and_text_models = ['clip']

# Apple support 
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default to cpu for now
#Mdevice = 'cpu'
MAX_IMG_SAMPLES = 1000

def fit_model(model_to_fit, data):
    # TEXT
    if model_to_fit == 't5_small':
        model_name = 't5-small'
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_to_fit == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2').to(device)
    elif model_to_fit == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartModel.from_pretrained('facebook/bart-base').to(device)
    elif model_to_fit == 'pegasus':
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
    elif model_to_fit == 'distil_bert':
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_to_fit == 'clip_text':
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # IMAGES
    elif model_to_fit == 'vit_base':
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    elif model_to_fit == 'deit_small':
        feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224')
        model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224').to(device)
    elif model_to_fit == 'resnet':
        model = models.resnet50(pretrained=True).to(device)
        # Remove the last fully-connected layer (the classifier)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_to_fit == 'clip_image':
        feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # TEXT AND IMAGES
    elif model_to_fit == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError("Invalid model_to_fit value. Choose from ['t5_small', 'albert', 'bart', 'pegasus', 'distil_bert']")

    if model_to_fit in text_models:
        # preprocessing step
        data["review"] = data["review"].apply(preprocess_text)
        experiment_ids = data["experiment_id"].values
        reviews = data["review"].values
    elif model_to_fit in image_models:
        experiment_ids = data["experiment_id"].values
        image_files = data["image"].apply(lambda x: x.replace('/p/', '') if pd.notnull(x) else x).values

    unique_experiment_ids = data["experiment_id"].unique()
    vintage_embeddings = {}

    for experiment_id in unique_experiment_ids:
        idx = (experiment_ids == experiment_id)
        if model_to_fit in text_models:
            vintage_reviews = reviews[idx]
            vintage_embedding = create_vintage_embedding_text(model=model,
                                                              vintage_ids=[experiment_id] * len(vintage_reviews),
                                                              reviews=vintage_reviews,
                                                              tokenizer=tokenizer,
                                                              device=device).mean(axis=0)
        elif model_to_fit in image_models:
            vintage_image_files = image_files[idx]
            try:
                vintage_image_files = random.sample(list(vintage_image_files), MAX_IMG_SAMPLES)
            except: 
                vintage_image_files = list(vintage_image_files)
            vintage_embedding = download_process_delete_images(urls=image_files, model=model, feature_extractor=feature_extractor, device=device).mean(axis=0)
            vintage_embedding = vintage_embedding.detach().numpy()
        elif model_to_fit == 'clip':
            # For text
            reviews = reviews[idx]
            inputs_text = processor(text=reviews, return_tensors="pt", padding=True, truncation=True)
            outputs_text = model.get_text_features(**inputs_text)
            vintage_text_embedding = outputs_text.detach().numpy().mean(axis=0)
            # For images
            image_files = image_files[idx]
            try:
                image_files = random.sample(list(image_files), MAX_IMG_SAMPLES)
            except: 
                image_files = list(image_files)
            # Use your existing function for image processing
            images = download_process_delete_images(urls=image_files, model=model, feature_extractor=feature_extractor, device=device).mean(axis=0)
            inputs_image = processor(images=images, return_tensors="pt")
            outputs_image = model.get_image_features(**inputs_image)
            vintage_image_embedding = outputs_image.detach().numpy().mean(axis=0)
            # Combine embeddings
            vintage_embedding = (vintage_text_embedding + vintage_image_embedding) / 2

        vintage_embeddings[experiment_id] = vintage_embedding

    embedding_matrix =  np.array(list(vintage_embeddings.values()))
    return embedding_matrix, unique_experiment_ids

def download_process_delete_images(urls, model, feature_extractor, device, batch_size=10):
    embeddings = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        images = []
        for url in batch_urls:
            if pd.isnull(url):
                images.append(torch.zeros((224, 224, 3)))  # Assuming input size for the model is 224x224x3
                continue
            url = "https://images.vivino.com/labels/" + url
            url = url.replace('p/', '')
            response = requests.get(url)
            try:
                img = Image.open(BytesIO(response.content))
                images.append(F.to_tensor(img))
            except PIL.UnidentifiedImageError:
                images.append(torch.zeros((224, 224, 3)))  # Assuming input size for the model is 224x224x3

        # Preprocess the images with ViTFeatureExtractor
        inputs = feature_extractor(images=images, return_tensors="pt")
        inputs = inputs.to(device)  # send inputs to the same device as model
        outputs = model(**inputs)
        features = outputs.logits

        # Flatten the tensor to 2D and then take the mean
        cls_embeddings = features.view(features.size(0), -1).mean(dim=0, keepdim=True)
        embeddings.append(cls_embeddings)

    embeddings = torch.cat(embeddings, dim=0)  # Concatenate along the first dimension
    embeddings = embeddings.view(embeddings.size(0), -1) # Reshape to 2D tensor
    return embeddings


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

def create_vintage_embedding_text(model, vintage_ids, reviews, device, tokenizer=None, batch_size=10):
    embeddings = []
    for i in range(0, len(vintage_ids), batch_size):
        batch_vintage_ids = vintage_ids[i:i+batch_size]
        batch_reviews = reviews[i:i+batch_size]
        input_text = [f"vintage {vintage_id}: {review}" for vintage_id, review in zip(batch_vintage_ids, batch_reviews)]
        input_tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # Move the tokenized input to the correct device
        input_tokenized = {k: v.to(device) for k, v in input_tokenized.items()}
        with torch.no_grad():
            output = model(**input_tokenized)
        batch_embeddings = output.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)