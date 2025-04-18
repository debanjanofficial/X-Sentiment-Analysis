import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TwitterDataset(Dataset):
    def __init__(self, tweets, labels=None, tokenizer=None, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        encoding = self.tokenizer(tweet, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_data(csv_path, tokenizer=None, max_length=128, batch_size=16, has_labels=True):
    # Read CSV without headers
    df = pd.read_csv(csv_path, header=None)
    
    if has_labels:
        # Format: tweet_id, sentiment, tweet
        tweet_ids = df.iloc[:, 0].tolist()
        sentiments = df.iloc[:, 1].tolist()
        tweets = df.iloc[:, 2].tolist()
        
        dataset = TwitterDataset(tweets, sentiments, tokenizer, max_length)
    else:
        # Format: tweet_id, tweet
        tweet_ids = df.iloc[:, 0].tolist()
        tweets = df.iloc[:, 1].tolist()
        
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
