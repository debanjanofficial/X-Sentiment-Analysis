import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Trainer:
    
    def __init__(self, model, train_dataloader, val_dataloader=None, output_dir="outputs"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self, epochs=3, learning_rate=2e-5):
        return self.model.train(
            self.train_dataloader,
            self.val_dataloader,
            epochs,
            learning_rate
        )
    
    def evaluate(self, dataloader=None):
        if dataloader is None:
            if self.val_dataloader is None:
                raise ValueError("No validation dataloader provided")
            dataloader = self.val_dataloader
        
        # Get predictions
        results = self.model.predict(dataloader)
        predictions = results['predictions']
        
        # Get true labels
        true_labels = []
        for batch in dataloader:
            if 'labels' in batch:
                true_labels.extend(batch['labels'].numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Print metrics
        print(f"\nEvaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.output_dir, "model")
        
        self.model.save(path)
