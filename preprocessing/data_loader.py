import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TwitterDataset(Dataset):
    def __init__(self, tweets, labels=None, tokenizer=None, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiment_mapping = {"Positive": 1, "Negative": 0}
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        
        if not isinstance(tweet, str):
            if tweet is None or pd.isna(tweet):
                tweet = ""  # Empty string for None/NaN values
            else:
                tweet = str(tweet)
            
        encoding = self.tokenizer(tweet, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()}
        if self.labels is not None:
            if isinstance(self.labels[idx], str):
                label = self.sentiment_mapping.get(self.labels[idx], 0)
            else:
                label = int(self.labels[idx])
            item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def load_data(csv_path, tokenizer=None, max_length=128, batch_size=16, has_labels=True):
    # Read CSV without headers
    df = pd.read_csv(csv_path, header=None)
    
    if has_labels:
        # Format: tweet_id, category, sentiment, tweet
        tweet_ids = df.iloc[:, 0].tolist()
        categories = df.iloc[:, 1].tolist()
        sentiment_mapping = {"Positive": 1, "Negative": 0}
        sentiments = [sentiment_mapping.get(label, 0) for label in df.iloc[:, 2].tolist()]
        tweets = df.iloc[:, 3].tolist()
        
        for i, tweet in enumerate(tweets[:5]):
            print(f"Tweet {i} type: {type(tweet)}, preview: {tweet[:50] if isinstance(tweet, str) else tweet}")
        
        
        dataset = TwitterDataset(tweets, sentiments, tokenizer, max_length)
    else:
        # Format: tweet_id, tweet
        tweet_ids = df.iloc[:, 0].tolist()
        tweets = df.iloc[:, 3].tolist()
        
        dataset = TwitterDataset(tweets, None, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=has_labels  # Only shuffle for training
    )
    
    return dataloader, tweet_ids

def load_word_lists(positive_path, negative_path):
    with open(positive_path, 'r', encoding='utf-8') as f:
        positive_words = set(line.strip() for line in f if line.strip() and not line.startswith(';'))
    
    with open(negative_path, 'r', encoding='utf-8') as f:
        negative_words = set(line.strip() for line in f if line.strip() and not line.startswith(';'))
    
    return positive_words, negative_words
