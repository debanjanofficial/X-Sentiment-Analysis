import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from models.base_model import BaseModel

class HuggingFaceModel(BaseModel):
    """Hugging Face model for sentiment analysis."""
    
    def __init__(self, model_name, num_labels=2, use_mps=True):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Set device (MPS for M1 Pro)
        if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            print(f"Using MPS device for {model_name}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {self.device} device for {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
    
    def train(self, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5):
        """Train the model."""
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc="Training")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_dataloader)
            history['train_loss'].append(avg_train_loss)
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss, val_accuracy = self._evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"Validation loss: {val_loss:.4f}")
                print(f"Validation accuracy: {val_accuracy:.4f}")
        
        return history
    
    def _evaluate(self, dataloader):
        """Evaluate the model on validation data."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()
                
                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_val_loss, accuracy
    
    def predict(self, dataloader):
        """Make predictions on test data."""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Predicting with {self.model_name}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Move to CPU and convert to numpy
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities)
        }
    
    def save(self, path):
        """Save the model."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
