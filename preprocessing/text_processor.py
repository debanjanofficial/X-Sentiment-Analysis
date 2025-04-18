import re
import nltk

class TextProcessor:
    def __init__(self, pos_words=None, neg_words=None):
        self.pos_words = set(pos_words) if pos_words else set()
        self.neg_words = set(neg_words) if neg_words else set()
        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)    # Remove mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        tokens = nltk.word_tokenize(text)
        # Join tokens back into text
        return ' '.join(tokens)
    
    def extract_features(self, text):
        tokens = text.split()
        pos_count = sum(1 for word in tokens if word in self.pos_words)
        neg_count = sum(1 for word in tokens if word in self.neg_words)
        sentiment_score = pos_count - neg_count
        return {
            'pos_count': pos_count, 
            'neg_count': neg_count,
            'sentiment_score': sentiment_score
        }
