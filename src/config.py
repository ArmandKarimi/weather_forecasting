
import os

#---------- SET LOGGING ------------------------
# Get the absolute path of the project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Moves one level up
# Define the logs directory path relative to the project root
LOG_DIR = os.path.join(BASE_DIR, "output", "logs")  # Ensures logs are saved in 'output/logs'
# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --------- MODEL PARAMS ------------
#input sequence config
SEQ_LENGTH = 24
PRED_LENGTH = 1
BATCH_SIZE = 32

#model
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001

#train & eval
EPOCHS = 30