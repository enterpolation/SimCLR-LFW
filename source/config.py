import torch

IMAGE_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.2
NUM_EPOCH = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
