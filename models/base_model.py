from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for sentiment analysis models."""
    
    @abstractmethod
    def train(self, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, dataloader):
        """Make predictions."""
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the model."""
        pass
