import torch


ORIGINAL_SIZE = 255  # original image size
IMAGE_SIZE = 64  # augmented image size
BATCH_SIZE = 128
LEARNING_RATE = 0.1
NUM_EPOCH = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
