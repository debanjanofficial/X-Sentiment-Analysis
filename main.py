import os
import argparse
import torch
from transformers import AutoTokenizer

# Import project modules
import config
from preprocessing.data_loader import load_data, load_word_lists
from preprocessing.text_processor import TextProcessor
from models.huggingface_models import HuggingFaceModel
from models.ensemble_model import VotingEnsemble, StackingEnsemble
from training.trainer import Trainer
from inference.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--model_path", type=str, default="outputs/model", help="Model path")
    parser.add_argument("--output_path", type=str, default="outputs/predictions.csv", help="Predictions path")
    args = parser.parse_args()
    
    # Check MPS availability for M1 Pro
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device (M1 Pro GPU)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")
    
    # Load word lists
    print("Loading word lists...")
    positive_words, negative_words = load_word_lists(
        config.POSITIVE_WORDS_PATH,
        config.NEGATIVE_WORDS_PATH
    )
    
    # Initialize text processor
    text_processor = TextProcessor(positive_words, negative_words)
    
    # Initialize models
    models = []
    for model_name in config.PRETRAINED_MODELS:
        print(f"Initializing model: {model_name}")
        model = HuggingFaceModel(
            model_name=model_name,
            num_labels=2,
            use_mps=config.USE_MPS
        )
        models.append(model)
    
    # Create ensemble
    ensemble = VotingEnsemble(models, weights=config.ENSEMBLE_WEIGHTS)
    
    # Training
    if args.train:
        print("Loading training data...")
        train_dataloader, _ = load_data(
            config.TRAIN_DATA_PATH,
            tokenizer=models[0].tokenizer,  # Use the first model's tokenizer
            max_length=config.MAX_LENGTH,
            batch_size=config.BATCH_SIZE,
            has_labels=True
        )
        
        print("Loading validation data...")
        val_dataloader, _ = load_data(
            config.VALIDATION_DATA_PATH,
            tokenizer=models[0].tokenizer,
            max_length=config.MAX_LENGTH,
            batch_size=config.BATCH_SIZE,
            has_labels=True
        )
        
        # Train ensemble
        trainer = Trainer(
            model=ensemble,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir="outputs"
        )
        
        print("Starting training...")
        trainer.train(epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE)
        
        # Evaluate ensemble
        print("Evaluating ensemble...")
        trainer.evaluate()
        
        # Save model
        print(f"Saving model to {args.model_path}...")
        trainer.save_model(args.model_path)
    
    # Prediction
    if args.predict:
        if not args.train:
            # Load model if not trained
            print(f"Loading model from {args.model_path}...")
            ensemble.load(args.model_path)
        
        print("Loading test data...")
        test_dataloader, tweet_ids = load_data(
            config.VALIDATION_DATA_PATH,  # In a real scenario, this would be test data
            tokenizer=models[0].tokenizer,
            max_length=config.MAX_LENGTH,
            batch_size=config.BATCH_SIZE,
            has_labels=False
        )
        
        # Make predictions
        predictor = Predictor(
            model=ensemble,
            dataloader=test_dataloader,
            tweet_ids=tweet_ids
        )
        
        print("Making predictions...")
        results = predictor.predict()
        
        # Save predictions
        print(f"Saving predictions to {args.output_path}...")
        predictor.save_predictions(results['predictions'], args.output_path)

if __name__ == "__main__":
    main()
