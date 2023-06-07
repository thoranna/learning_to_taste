import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_torch_and_device(triplets):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplets = torch.tensor(triplets).to(device)
    return triplets