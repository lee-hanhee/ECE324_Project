import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from model import create_model
from dataset import InstrumentDataset
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import KFold

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer, scheduler: ReduceLROnPlateau,
               early_stopping: EarlyStopping, device: torch.device, fold: int) -> Tuple[float, float]:
    """Train a single fold of the cross-validation."""
    best_val_loss = float('inf')
    
    for epoch in range(50):  # Max epochs per fold
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Fold {fold} - Epoch {epoch+1}/50 - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Fold {fold} - Epoch {epoch+1}/50 - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Fold {fold} - Epoch {epoch+1}/50:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Early stopping
        early_stopping(val_loss, model, f'best_model_fold_{fold}.pth')
        if early_stopping.early_stop:
            print(f'Early stopping triggered for fold {fold}')
            break
    
    return val_loss, val_acc

def ensemble_predict(models: List[nn.Module], test_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Make predictions using an ensemble of models."""
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(torch.sigmoid(outputs))
        
        all_predictions.append(torch.cat(predictions, dim=0))
    
    # Average predictions from all models
    ensemble_predictions = torch.stack(all_predictions).mean(dim=0)
    return ensemble_predictions

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    data_dir = os.path.join(project_root, 'data', 'raw')
    
    print(f"Loading data from: {data_dir}")
    
    # Create dataset
    dataset = InstrumentDataset(
        root_dir=data_dir,
        transform=None,
        sample_rate=44100,
        n_mels=128
    )
    
    # Initialize cross-validation
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store models and their validation scores
    models = []
    fold_scores = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\nFOLD {fold + 1}/{n_splits}')
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_subsampler,
            num_workers=4
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=val_subsampler,
            num_workers=4
        )
        
        # Create model for this fold
        model = create_model(num_classes=len(dataset.class_to_idx))
        model = model.to(device)
        
        # Define loss function, optimizer, scheduler, and early stopping
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        
        # Train the model
        val_loss, val_acc = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            device=device,
            fold=fold + 1
        )
        
        # Store model and score
        models.append(model)
        fold_scores.append(val_acc)
        
        print(f'Fold {fold + 1} Validation Accuracy: {val_acc:.4f}')
    
    # Print average cross-validation score
    print(f'\nAverage Cross-Validation Accuracy: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})')
    
    # Create test set for ensemble evaluation
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Make ensemble predictions
    ensemble_predictions = ensemble_predict(models, test_loader, device)
    
    # Calculate ensemble accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (_, labels), preds in zip(test_loader, ensemble_predictions):
            labels = labels.to(device)
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    
    ensemble_acc = correct / total
    print(f'\nEnsemble Test Accuracy: {ensemble_acc:.4f}')
    
    # Save ensemble models
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'ensemble_model_{i+1}.pth')

if __name__ == '__main__':
    main() 