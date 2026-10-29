import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import load_and_preprocess_magnet_data, MagNetDataset
from src.model import PhysicsTCNLSTM
from src.loss import PhysicsGuidedDstLoss
from src.evaluate import evaluate_no_history_models

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Using compute device: {device}")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'best_dst_model.pt')

    # 1. Load preprocessed dataframe
    df_magnet = load_and_preprocess_magnet_data()

    # 2. Chronological Split (70% Train / 10% Val / 20% Test)
    train_end = int(len(df_magnet) * 0.7)
    val_end = int(len(df_magnet) * 0.8)

    train_df = df_magnet.iloc[:train_end].copy()
    val_df = df_magnet.iloc[train_end:val_end].copy()
    test_df = df_magnet.iloc[val_end:].copy()

    # 3. Instantiate Datasets
    seq_len = 48
    forecast_horizon = 24
    train_dataset = MagNetDataset(train_df, seq_len=seq_len, forecast_horizon=forecast_horizon, scaler=None, fit_scaler=True)
    val_dataset = MagNetDataset(val_df, seq_len=seq_len, forecast_horizon=forecast_horizon, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = MagNetDataset(test_df, seq_len=seq_len, forecast_horizon=forecast_horizon, scaler=train_dataset.scaler, fit_scaler=False)

    # 4. Weighted Sampler for Storm Events (Cleaned)
    y_tensor = train_dataset.y
    train_dst_targets = (y_tensor[:, 0] if y_tensor.ndim > 1 else y_tensor).numpy()
    train_dst_targets = np.nan_to_num(train_dst_targets, nan=0.0)
    
    sample_weights = np.where(train_dst_targets < -30.0, 6.0, 1.0)
    sample_weights = np.nan_to_num(sample_weights, nan=1.0)
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # 5. DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders initialized with {len(train_dataset.feature_cols)} engineered physical features.")

    # 6. Model Initialization
    num_features = len(train_dataset.feature_cols)
    model = PhysicsTCNLSTM(
        input_dim=num_features,
        num_channels=[16, 32, 64],
        lstm_hidden=64,
        forecast_horizon=forecast_horizon,
        dropout=0.2
    ).to(device)

    # 7. Physics-Guided Loss & Optimizer
    criterion = PhysicsGuidedDstLoss(alpha=4.0, lambda_phys=0.05, threshold=-30.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 8. Training Loop
    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            best_marker = f" [Saved Best Model -> {checkpoint_path}]"
        else:
            best_marker = ""

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{best_marker}")

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")

    # 9. Model Evaluation
    evaluate_no_history_models(
        model, 
        test_loader, 
        test_df=test_df,
        checkpoint_path=checkpoint_path, 
        output_dir=output_dir,
        device=device
    )

if __name__ == '__main__':
    main()