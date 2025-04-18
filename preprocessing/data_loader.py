import pandas as pd
from torch.utils.data import Dataset

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
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

def load_data(train_path, test_path=None, pos_words_path=None, neg_words_path=None):
    # Load training data
    train_df = pd.read_csv(train_path, header=None, names=['tweet_id', 'sentiment', 'tweet'])
    # Remove header if present
    if train_df.iloc[0]['tweet_id'] == 'tweet_id': # Fixed indexing error
        train_df = train_df.iloc[1:]
    train_df['sentiment'] = train_df['sentiment'].astype(int)
    
    # Load test data
    test_df = None
    if test_path:
        test_df = pd.read_csv(test_path, header=None, names=['tweet_id', 'tweet'])
        if test_df.iloc[0]['tweet_id'] == 'tweet_id': # Fixed indexing error
            test_df = test_df.iloc[1:]
    
    # Load positive and negative words
    pos_words = []
    neg_words = []
    if pos_words_path:
        with open(pos_words_path, 'r') as f:
            pos_words = [line.strip() for line in f if line.strip() and not line.startswith(';')]
    if neg_words_path:
        with open(neg_words_path, 'r') as f:
            neg_words = [line.strip() for line in f if line.strip() and not line.startswith(';')]
    
    return train_df, test_df, pos_words, neg_words
