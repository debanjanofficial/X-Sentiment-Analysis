import numpy as np
from models.base_model import BaseModel

class VotingEnsemble(BaseModel):
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1] * len(models)
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def train(self, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5):
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.model_name}")
            history = model.train(train_dataloader, val_dataloader, epochs, learning_rate)
            histories.append(history)
        
        return histories
    
    def predict(self, dataloader):
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            print(f"Getting predictions from model {i+1}/{len(self.models)}")
            result = model.predict(dataloader)
            all_predictions.append(result['predictions'])
            all_probabilities.append(result['probabilities'])
        
        # Apply weighted voting to probabilities
        weighted_probs = np.zeros_like(all_probabilities[0])
        for i, probs in enumerate(all_probabilities):
            weighted_probs += self.weights[i] * probs
        
        # Get final predictions
        ensemble_predictions = np.argmax(weighted_probs, axis=1)
        
        return {
            'predictions': ensemble_predictions,
            'probabilities': weighted_probs
        }
    
    def save(self, path):
        for i, model in enumerate(self.models):
            model_path = f"{path}_model{i+1}"
            model.save(model_path)
    
    def load(self, path):
        for i, model in enumerate(self.models):
            model_path = f"{path}_model{i+1}"
            model.load(model_path)

class StackingEnsemble(BaseModel):
    
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
    
    def train(self, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5):
        # Train base models
        for i, model in enumerate(self.base_models):
            print(f"\nTraining base model {i+1}/{len(self.base_models)}: {model.model_name}")
            model.train(train_dataloader, val_dataloader, epochs, learning_rate)
        
        # Generate meta-features and train meta-model
        # This is a simplified implementation - in a full project, this would be more complex
        print("\nTraining meta-model...")
        self.meta_model.train(train_dataloader, val_dataloader, epochs, learning_rate)
        
        return {"message": "Ensemble training completed"}
    
    def predict(self, dataloader):
        # Get base model predictions
        base_probabilities = []
        
        for i, model in enumerate(self.base_models):
            print(f"Getting predictions from base model {i+1}/{len(self.base_models)}")
            result = model.predict(dataloader)
            base_probabilities.append(result['probabilities'])
        
        # For simplicity, we'll use a weighted average of base model predictions
        # In a full implementation, the meta-model would use these predictions as features
        weighted_probs = np.mean(base_probabilities, axis=0)
        ensemble_predictions = np.argmax(weighted_probs, axis=1)
        
        return {
            'predictions': ensemble_predictions,
            'probabilities': weighted_probs
        }
    
    def save(self, path):
        for i, model in enumerate(self.base_models):
            model_path = f"{path}_base{i+1}"
            model.save(model_path)
        
        meta_model_path = f"{path}_meta"
        self.meta_model.save(meta_model_path)
    
    def load(self, path):
        for i, model in enumerate(self.base_models):
            model_path = f"{path}_base{i+1}"
            model.load(model_path)
        
        meta_model_path = f"{path}_meta"
        self.meta_model.load(meta_model_path)
