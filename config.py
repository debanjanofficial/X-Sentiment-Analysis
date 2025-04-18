# Data paths
TRAIN_DATA_PATH = "data/raw/twitter_train.csv"
VALIDATION_DATA_PATH = "data/raw/twitter_validation.csv"
POSITIVE_WORDS_PATH = "data/raw/positive-words.txt"
NEGATIVE_WORDS_PATH = "data/raw/negative-words.txt"

# Model settings
PRETRAINED_MODELS = [
    "cardiffnlp/twitter-roberta-base-sentiment",  # Twitter-specific RoBERTa
    "finiteautomata/bertweet-base-sentiment-analysis",  # BERTweet for Twitter
    "distilbert-base-uncased"  # Smaller general model
]

# Training settings
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 3
LEARNING_RATE = 2e-5
VALIDATION_SPLIT = 0.2

# Ensemble settings
ENSEMBLE_METHOD = "voting" 
ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]  # Weights for each model

# Device settings
USE_MPS = True 
