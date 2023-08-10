import torch
import numpy as np
from transformers import T5EncoderModel, AutoTokenizer, AlbertTokenizer, AlbertModel, \
    BartTokenizer, BartModel, AutoModel, CLIPModel, AutoImageProcessor, ViTModel, DeiTModel, ResNetModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, CLIPTextModel
from transformers import AutoProcessor, CLIPVisionModel
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartModel
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
import h5py
from transformers import set_seed
from utils.set_seed import RANDOM_SEED

# You might need to download the stopwords first
nltk.download('punkt')
nltk.download('stopwords')

text_models = ['t5_small', 'albert', 'bart', 'distil_bert', 'clip_text', 'flan_t5', 'pegasus', 'bart_large']
image_models = ['vit_base', 'deit_small', 'resnet', 'clip_image']
image_and_text_models = ['clip']

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

# Default to cpu
device = 'cpu'

def fit_model(model_to_fit, data):
    set_deterministic(RANDOM_SEED)
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
    elif model_to_fit == 'distil_bert':
        model_name = "distilbert-base-uncased"
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_to_fit == 'clip_text':
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    elif model_to_fit == 'flan_t5':
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    elif model_to_fit == 'pegasus':
        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    elif model_to_fit == 'bart_large':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartModel.from_pretrained('facebook/bart-large')

    # IMAGES
    elif model_to_fit == 'vit_base':
        feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    elif model_to_fit == 'deit_small':
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224").to(device)
    elif model_to_fit == 'resnet':
        model = ResNetModel.from_pretrained("microsoft/resnet-50")
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    elif model_to_fit == 'clip_image':
        feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # TEXT AND IMAGES
    elif model_to_fit == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError("Invalid model_to_fit value. Choose from ['t5_small', 'albert', 'bart', 'distil_bert']")

    if model_to_fit in text_models:
        # preprocessing step
        data["review"] = data["review"].apply(preprocess_text)
        reviews = data["review"].values
    elif model_to_fit in image_models:
        image_files = data["image"].apply(lambda x: x.replace('/p/', '') if pd.notnull(x) else x).values
    else:
        data["review"] = data["review"].apply(preprocess_text)
        reviews = data["review"].values
        image_files = data["image"].apply(lambda x: x.replace('/p/', '') if pd.notnull(x) else x).values
    experiment_ids = data["experiment_id"].values
    unique_experiment_ids = data["experiment_id"].unique()

    # If 'embeddings' dataset exists, delete it

    with h5py.File('embeddings.hdf5', 'a') as hf:
        if "mean_embeddings" in hf.keys():
            del hf["mean_embeddings"]
        for i, experiment_id in enumerate(unique_experiment_ids):
            if "embeddings" in hf.keys():
                del hf["embeddings"]
            print("experiment id: ", experiment_id)
            print("i: ", i)
            print("total: ", len(unique_experiment_ids))
            idx = (experiment_ids == experiment_id)
            if model_to_fit in text_models:
                vintage_reviews = reviews[idx]
                vintage_embedding = create_vintage_embedding_text(model=model,
                                                                  vintage_ids=[experiment_id] * len(vintage_reviews),
                                                                  reviews=vintage_reviews,
                                                                  tokenizer=tokenizer,
                                                                  device=device).mean(axis=0)
                try:
                    if model_to_fit != 'resnet':
                        hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1), axis=0)
                        hf["mean_embeddings"][-1:] = vintage_embedding
                    else:
                        hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1, 1, 7, 7))
                        hf["mean_embeddings"][-1, :, :, :] = vintage_embedding
                except KeyError:
                    if model_to_fit != 'resnet':
                        hf.create_dataset("mean_embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))
                    else:
                        hf.create_dataset("mean_embeddings", data=np.expand_dims(vintage_embedding, axis=0), maxshape=(None, 1, 7, 7))
            elif model_to_fit in image_models:
                vintage_image_files = image_files[idx]
                try:
                    MAX_IMG_SAMPLES = 100
                    vintage_image_files = random.sample(list(vintage_image_files), MAX_IMG_SAMPLES)
                except: 
                    vintage_image_files = list(vintage_image_files)
                download_process_delete_images(urls=vintage_image_files, model=model, model_name=model_to_fit, feature_extractor=feature_extractor, device=device)
                # Load embeddings from the file
                if model_to_fit == 'clip_image':
                    embeddings = hf['clip_embeddings'][:]
                else:
                    embeddings = hf['embeddings'][:]  # This is a reference to the data on disk
                vintage_embedding = embeddings.mean(axis=0)
                if model_to_fit == 'clip_image':
                    try:
                        if model_to_fit != 'resnet':
                            hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1), axis=0)
                            hf["mean_embeddings"][-1:] = vintage_embedding
                        else:
                            hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1, 1, 7, 7))
                            hf["mean_embeddings"][-1, :, :, :] = vintage_embedding
                    except KeyError:
                        if model_to_fit != 'resnet':
                            hf.create_dataset("mean_embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))
                        else:
                            hf.create_dataset("mean_embeddings", data=np.expand_dims(vintage_embedding, axis=0), maxshape=(None, 1, 7, 7))
                else:
                    try:
                        if model_to_fit != 'resnet':
                            hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1), axis=0)
                            hf["mean_embeddings"][-1:] = vintage_embedding
                        else:
                            hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1, 1, 7, 7))
                            hf["mean_embeddings"][-1, :, :, :] = vintage_embedding
                    except KeyError:
                        if model_to_fit != 'resnet':
                            hf.create_dataset("mean_embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))
                        else:
                            hf.create_dataset("mean_embeddings", data=np.expand_dims(vintage_embedding, axis=0), maxshape=(None, 1, 7, 7))
                    # vintage_embedding = vintage_embedding.detach().numpy()
            elif model_to_fit == 'clip':
                # For text
                vintage_reviews = reviews[idx]
                vintage_text_embedding = create_vintage_embedding_text(
                    model=model, vintage_ids=[experiment_id] * len(vintage_reviews),
                    reviews=vintage_reviews,
                    tokenizer=tokenizer,
                    device=device).mean(axis=0)
                # For images
                try:
                    MAX_IMG_SAMPLES = 100
                    vintage_image_files = random.sample(list(vintage_image_files), MAX_IMG_SAMPLES)
                except: 
                    vintage_image_files = list(vintage_image_files)
                # Use your existing function for image processing
                download_process_delete_images(urls=vintage_image_files, model=model, model_name=model_to_fit, feature_extractor=feature_extractor, device=device)
                # Load embeddings from the file
                embeddings = hf['embeddings'][:]  # This is a reference to the data on disk
                vintage_image_embedding = embeddings.mean(axis=0)
                # Combine embeddings
                vintage_embedding = (vintage_text_embedding + vintage_image_embedding) / 2
                try:
                    if model_to_fit != 'resnet':
                        hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1), axis=0)
                        hf["mean_embeddings"][-1:] = vintage_embedding
                    else:
                        hf["mean_embeddings"].resize((hf["mean_embeddings"].shape[0] + 1, 1, 7, 7))
                        hf["mean_embeddings"][-1, :, :, :] = vintage_embedding
                except KeyError:
                    # If 'mean_embeddings' dataset doesn't exist, create it
                    if model_to_fit != 'resnet':
                        hf.create_dataset("mean_embeddings", data=[vintage_embedding], maxshape=(None, vintage_embedding.shape[0]))
                    else:
                        hf.create_dataset("mean_embeddings", data=np.expand_dims(vintage_embedding, axis=0), maxshape=(None, 1, 7, 7))
    
        # At the end, load all embeddings at once as a memmap array
        embedding_matrix = hf["mean_embeddings"][:]
        if model_to_fit == 'resnet':
            embedding_matrix = embedding_matrix.reshape((embedding_matrix.shape[0], -1))
        return embedding_matrix, unique_experiment_ids


def download_process_delete_images(urls, model, model_name, device, feature_extractor, batch_size=10, embedding_dim=768, embedding_path='embeddings.hdf5'):
    with h5py.File(embedding_path, 'a') as hf:  
        if "embeddings" not in hf:
            if model_name != 'resnet':
                hf.create_dataset("embeddings", data=np.empty((0, embedding_dim)), maxshape=(None, embedding_dim))
            else:
                hf.create_dataset("embeddings", data=np.empty((0, 1, 7, 7)), maxshape=(None, 1, 7, 7))
        if "clip_embeddings" not in hf:
            hf.create_dataset("clip_embeddings", data=np.empty((0, embedding_dim)), maxshape=(None, embedding_dim))

        for batch_start in range(0, len(urls), batch_size):
            batch_urls = urls[batch_start: batch_start + batch_size]
            for url in batch_urls:
                if pd.isnull(url):
                    if model_name != 'resnet':
                        emb = torch.zeros((1, embedding_dim)).to(device)
                    else:
                        emb =torch.zeros((1, 1, 7, 7)).to(device)
                else:
                    url = "https://images.vivino.com/labels/" + url
                    url = url.replace('p/', '')
                    response = requests.get(url)
                    try:
                        img = Image.open(BytesIO(response.content))
                    except Exception:  # Catch all exceptions, not just UnidentifiedImageError
                        if model_name != 'resnet':
                            emb = torch.zeros((1, embedding_dim)).to(device)
                        else:
                            emb =torch.zeros((1, 1, 7, 7)).to(device)
                    else:
                        preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
                        if model_name == 'clip_image':
                            outputs = model(img_tensor)
                            emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                        elif model_name == 'resnet':
                            outputs = model(img_tensor)
                            # emb = outputs.view(outputs.size(0), -1).mean(dim=0, keepdim=True)
                            emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                        else:
                            inputs = feature_extractor(images=F.to_tensor(img), return_tensors="pt")
                            inputs = inputs.to(device)  # send inputs to the same device as model
                            outputs = model(**inputs)
                            emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()

                # Convert tensor to numpy and append it to the HDF5 file.
                if model_name == 'clip' or model_name == 'clip_image':
                    emb_numpy = emb
                    new_shape = (hf["clip_embeddings"].shape[0] + emb_numpy.shape[0], emb_numpy.shape[1])
                    hf["clip_embeddings"].resize(new_shape)
                    hf["clip_embeddings"][-emb_numpy.shape[0]:] = emb_numpy
                elif model_name == 'resnet':
                    emb_numpy = emb
                    hf["embeddings"].resize((hf["embeddings"].shape[0] + 1, 1, 7, 7))
                    hf["embeddings"][-1] = emb_numpy
                else:
                    emb_numpy = emb
                    new_shape = (hf["embeddings"].shape[0] + emb_numpy.shape[0], emb_numpy.shape[1])
                    hf["embeddings"].resize(new_shape)
                    hf["embeddings"][-emb_numpy.shape[0]:] = emb_numpy


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

def create_vintage_embedding_text(model, vintage_ids, reviews, device, tokenizer=None, batch_size=128):
    embeddings = []

    # Make sure model is in eval mode and on the right device
    # model.eval()
    # model.to(device)
    print("vintage ids: ", len(vintage_ids))
    print("num batches: ", len(list(range(0, len(vintage_ids), batch_size))))
    for i in range(0, len(vintage_ids), batch_size):
        print("batch: ", i)
        batch_vintage_ids = vintage_ids[i:i+batch_size]
        batch_reviews = reviews[i:i+batch_size]
        input_text = [f"vintage {vintage_id}: {review}" for vintage_id, review in zip(batch_vintage_ids, batch_reviews)]
        input_tokenized = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # Move the tokenized input to the correct device
        input_tokenized = {k: v.to(device) for k, v in input_tokenized.items()}

        with torch.no_grad():
            if model.__class__.__name__.lower() in ["pegasusforconditionalgeneration"]:  # Check if the model is of PEGASUS type
                # Bypass the decoder and only get encoder's output for PEGASUS and Flan-T5
                encoder_outputs = model.model.encoder(input_tokenized['input_ids'], attention_mask=input_tokenized['attention_mask'])
                batch_embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif model.__class__.__name__.lower() in ["t5forconditionalgeneration"]:
                # Bypass the decoder and only get encoder's output for PEGASUS and Flan-T5
                encoder_outputs = model.encoder(input_tokenized['input_ids'], attention_mask=input_tokenized['attention_mask'])
                batch_embeddings = encoder_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                # For other models
                output = model(**input_tokenized)
                batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)
