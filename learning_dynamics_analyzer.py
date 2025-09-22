#!/usr/bin/env python3
"""
Learning Dynamics Analyzer for MambaStock

This script analyzes training and validation losses to diagnose potential
underfitting or overfitting issues that may be causing prediction shrinkage.

It implements train/validation split and plots learning curves to help
determine if the model needs:
- More epochs (underfitting: both train and val loss high and close)
- Early stopping (overfitting: train << val loss)
- Regularization changes (weight decay, dropout)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from mambastock_model import MambaStock
from dataset import ClosePrice
from advanced_normalization import create_advanced_normalizer, apply_advanced_normalization
from torch.cuda.amp import autocast, GradScaler

def train_validation_split(dataset, val_ratio=0.2):
    """
    Split dataset into training and validation sets.
    
    For time series, we use the first 80% for training and last 20% for validation
    to maintain temporal order and avoid future information leakage.
    
    Args:
        dataset: The dataset object
        val_ratio: Fraction of data to use for validation
        
    Returns:
        train_data, val_data: Split tensors
    """
    total_timesteps = dataset.data.shape[1]
    train_end = int((1 - val_ratio) * total_timesteps)
    
    train_data = dataset.data[:, :train_end, :]
    val_data = dataset.data[:, train_end:, :]
    
    print(f"Data split: Train {train_data.shape[1]} timesteps, Val {val_data.shape[1]} timesteps")
    return train_data, val_data

def compute_loss_on_data(model, data_tensor, seq_len=90, pred_len=20, device=None, normalizer=None):
    """
    Compute average loss on a data tensor.
    
    Args:
        model: The model to evaluate
        data_tensor: Data tensor [n_stocks, n_timesteps, n_features]
        seq_len: Sequence length for model input
        pred_len: Prediction length for target calculation
        device: Device to run on
        normalizer: Advanced normalizer for consistent preprocessing
        
    Returns:
        average_loss: Float average loss across all valid windows
    """
    model.eval()
    
    n_stocks, n_timesteps, n_features = data_tensor.shape
    loss_fn = nn.MSELoss()
    
    # Calculate valid windows (need seq_len history + pred_len future)
    max_start_idx = n_timesteps - seq_len - pred_len
    if max_start_idx <= 0:
        return float('nan')  # Not enough data
    
    # Sample validation windows (every 10th window to speed up evaluation)
    eval_step = max(1, max_start_idx // 50)  # At most 50 evaluation points
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for start_idx in range(0, max_start_idx, eval_step):
            # Extract window and apply normalization
            window = data_tensor[:, start_idx:start_idx + seq_len, :].to(device)
            
            # Apply the same advanced normalization as training
            if normalizer is not None:
                input_norm = apply_advanced_normalization(window.unsqueeze(1), seq_len, normalizer).squeeze(1)
            else:
                # Fallback simple normalization
                mean = window.mean(dim=1, keepdim=True)
                std = window.std(dim=1, keepdim=True)
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                input_norm = (window - mean) / std
            
            # Compute targets: 20-minute log return
            eps = 1e-8
            p0 = data_tensor[:, start_idx + seq_len - 1, 3].to(device)  # Close at window end
            p1 = data_tensor[:, start_idx + seq_len + pred_len - 1, 3].to(device)  # Close 20 min later
            targets = (p1.add(eps).log() - p0.add(eps).log()).unsqueeze(-1)
            
            # Model prediction
            with autocast(enabled=(device.type == "cuda")):
                model_output = model(input_norm)
                if isinstance(model_output, tuple):
                    predictions = model_output[0]
                else:
                    predictions = model_output
                predictions = predictions.squeeze(-1)
            
            # Compute loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            total_samples += 1
    
    return total_loss / total_samples if total_samples > 0 else float('nan')

def analyze_learning_dynamics(h5_path="training_data.h5", 
                            seq_len=90, 
                            pred_len=20, 
                            num_epochs=10,
                            learning_rate=1e-4,
                            val_ratio=0.2):
    """
    Train model with validation tracking to analyze learning dynamics.
    
    Args:
        h5_path: Path to training data
        seq_len: Sequence length
        pred_len: Prediction length  
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        val_ratio: Validation split ratio
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ClosePrice(h5_path)
    print(f"Dataset loaded: {dataset.n_stocks} stocks, {dataset.data.shape}")
    
    # Train/validation split
    train_data, val_data = train_validation_split(dataset, val_ratio)
    
    # Initialize model
    input_size = dataset.data.shape[-1]
    model = MambaStock(input_size=input_size, seq_len=seq_len, pred_len=1).to(device)
    
    # Initialize advanced normalizer with training data only (no temporal leakage)
    print("Initializing advanced normalizer...")
    normalizer = create_advanced_normalizer(train_data, use_historical_only=True)
    
    # Load pretrained weights if available
    try:
        state_dict = torch.load("mambastock.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights from mambastock.pth")
    except FileNotFoundError:
        print("No pretrained weights found, starting from scratch")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    # Set up training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # Training tracking
    train_losses = []
    val_losses = []
    epochs_completed = []
    
    print(f"\nStarting learning dynamics analysis for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0
        
        # Sample training windows (every 5th window for speed)
        n_train_timesteps = train_data.shape[1]
        max_train_start = n_train_timesteps - seq_len - pred_len
        
        if max_train_start <= 0:
            print("❌ Not enough training data")
            break
            
        train_step = max(1, max_train_start // 100)  # At most 100 training steps per epoch
        
        for step_idx in range(0, max_train_start, train_step):
            # Extract training window
            window = train_data[:, step_idx:step_idx + seq_len, :].clone().to(device)
            
            # Apply advanced normalization
            input_norm = apply_advanced_normalization(window.unsqueeze(1), seq_len, normalizer).squeeze(1)
            
            # Add training noise for regularization
            noise_std = 0.01
            input_norm = input_norm + torch.randn_like(input_norm) * noise_std
            
            # Compute targets
            eps = 1e-8
            p0 = train_data[:, step_idx + seq_len - 1, 3].to(device)
            p1 = train_data[:, step_idx + seq_len + pred_len - 1, 3].to(device)
            targets = (p1.add(eps).log() - p0.add(eps).log()).unsqueeze(-1)
            
            # Forward pass
            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                model_output = model(input_norm)
                if isinstance(model_output, tuple):
                    predictions = model_output[0]
                else:
                    predictions = model_output
                predictions = predictions.squeeze(-1)
                loss = loss_fn(predictions, targets)
            
            # Backward pass
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            epoch_train_loss += loss.item()
            train_samples += 1
        
        avg_train_loss = epoch_train_loss / train_samples if train_samples > 0 else float('nan')
        
        # Validation phase
        print("  Computing validation loss...")
        avg_val_loss = compute_loss_on_data(model, val_data, seq_len, pred_len, device, normalizer)
        
        # Store results
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_completed.append(epoch + 1)
        
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Ratio:      {avg_val_loss/avg_train_loss:.2f}x" if avg_train_loss > 0 else "  Ratio: N/A")
        
        # Early stopping if validation loss increases significantly
        if len(val_losses) >= 3 and val_losses[-1] > val_losses[-3] * 1.5:
            print("  ⚠️  Validation loss increasing significantly, consider early stopping")
    
    print("\n" + "=" * 60)
    print("LEARNING DYNAMICS ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Analyze results
    final_train_loss = train_losses[-1] if train_losses else float('nan')
    final_val_loss = val_losses[-1] if val_losses else float('nan')
    
    print(f"\nFinal Results:")
    print(f"  Train Loss: {final_train_loss:.6f}")
    print(f"  Val Loss:   {final_val_loss:.6f}")
    
    if not np.isnan(final_train_loss) and not np.isnan(final_val_loss):
        ratio = final_val_loss / final_train_loss
        print(f"  Val/Train Ratio: {ratio:.2f}x")
        
        print(f"\nDiagnosis:")
        if ratio < 1.2 and final_train_loss > 0.01:
            print(f"  🔍 UNDERFITTING: Both losses high and close")
            print(f"     → Increase epochs, reduce weight decay, reduce noise augmentation")
        elif ratio > 2.0:
            print(f"  🔍 OVERFITTING: Train loss much lower than validation")
            print(f"     → Early stopping, increase regularization")
        elif ratio > 1.5:
            print(f"  🔍 MILD OVERFITTING: Some generalization gap")
            print(f"     → Monitor closely, consider slight regularization increase")
        else:
            print(f"  ✅ GOOD BALANCE: Training appears healthy")
    
    # Create plots
    print(f"\nCreating learning curves plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Learning curves
    plt.subplot(2, 2, 1)
    plt.plot(epochs_completed, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs_completed, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Loss ratio over time
    plt.subplot(2, 2, 2)
    ratios = [v/t if t > 0 else float('nan') for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs_completed, ratios, 'g-', linewidth=2)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect fit')
    plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Mild overfitting')
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Strong overfitting')
    plt.xlabel('Epoch')
    plt.ylabel('Validation / Training Loss')
    plt.title('Overfitting Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss improvement
    plt.subplot(2, 2, 3)
    if len(train_losses) > 1:
        train_improvements = [train_losses[0] - loss for loss in train_losses]
        val_improvements = [val_losses[0] - loss for loss in val_losses]
        plt.plot(epochs_completed, train_improvements, 'b-', linewidth=2, label='Train Improvement')
        plt.plot(epochs_completed, val_improvements, 'r-', linewidth=2, label='Val Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Improvement from Start')
    plt.title('Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss values (linear scale)
    plt.subplot(2, 2, 4)
    plt.plot(epochs_completed, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs_completed, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curves (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_dynamics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Learning curves saved to 'learning_dynamics_analysis.png'")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': epochs_completed,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'diagnosis': 'underfitting' if ratio < 1.2 and final_train_loss > 0.01 else 
                    ('overfitting' if ratio > 2.0 else 'balanced')
    }

if __name__ == "__main__":
    # Run learning dynamics analysis
    results = analyze_learning_dynamics(
        h5_path="training_data.h5",
        seq_len=90,
        pred_len=20,
        num_epochs=10,
        learning_rate=1e-4,
        val_ratio=0.2
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"   Diagnosis: {results['diagnosis']}")
    print(f"   Final train loss: {results['final_train_loss']:.6f}")
    print(f"   Final val loss: {results['final_val_loss']:.6f}")