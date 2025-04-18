import pandas as pd

class Predictor:
    
    def __init__(self, model, dataloader, tweet_ids=None):
        self.model = model
        self.dataloader = dataloader
        self.tweet_ids = tweet_ids
    
    def predict(self):
        return self.model.predict(self.dataloader)
    
    def save_predictions(self, predictions, output_path):
        if self.tweet_ids is None:
            raise ValueError("Tweet IDs not provided")
        
        if len(self.tweet_ids) != len(predictions):
            raise ValueError(f"Number of tweet IDs ({len(self.tweet_ids)}) does not match "
                             f"number of predictions ({len(predictions)})")
        
        # Create DataFrame with tweet_id and sentiment
        df = pd.DataFrame({
            'tweet_id': self.tweet_ids,
            'sentiment': predictions
        })
        
        # Save to CSV without header
        df.to_csv(output_path, index=False, header=False)
        
        print(f"Predictions saved to {output_path}")
