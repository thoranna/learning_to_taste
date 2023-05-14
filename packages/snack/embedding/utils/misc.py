import torch

def to_torch_and_device(triplets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplets = torch.tensor(triplets).to(device)
    return triplets