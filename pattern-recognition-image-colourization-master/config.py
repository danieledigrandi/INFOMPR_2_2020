import os

# DIRECTORY INFORMATION
DATASET = "Pictures"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "train"
TEST_DIR = "test"
VAL_DIR = "val"

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 32

# TRAINING INFORMATION
NUM_EPOCHS = 20
