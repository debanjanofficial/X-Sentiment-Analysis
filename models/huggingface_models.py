import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class HuggingFaceModel:
    def __init__(self, model_name, num_labels=2, device=None):
        # Set up device - use MPS if available for M1 Pro
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"Using MPS acceleration for {model_name}")
            else:
                self.device = torch.device("cpu")
                print(f"MPS not available, using CPU for {model_name}")
        else:
            self.device = device
            
        # Load tokenizer and model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
    
    def train(self, train_loader, val_loader=None, epochs=5, learning_rate=2e-5):
        # Set model to training mode
        self.model.train()
        
        # Set up optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                val_accuracy = self.evaluate(val_loader)
                print(f"Validation accuracy: {val_accuracy:.4f}")
        
        return self
    
    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                
        return correct / total
    
    def predict(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'tweet_id'}
                outputs = self.model(**batch)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities)
        }
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self
