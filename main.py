import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.dataset import load_and_preprocess_magnet_data, MagNetDataset
from src.model import PhysicsTCNLSTM
from src.loss import PhysicsGuidedDstLoss
from src.evaluate import evaluate_no_history_models
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="MagNet Geomagnetic Dst Index Forecasting Pipeline")
    parser.add_argument('--solar_wind_path' , type=str  , default='solar_wind.csv'   , help='Filename of solar wind CSV in data/')
    parser.add_argument('--labels_path'     , type=str  , default='labels.csv'       , help='Filename of labels CSV in data/')
    parser.add_argument('--sat_pos_path'    , type=str  , default='satellite_pos.csv', help='Filename of satellite position CSV in data/')
    parser.add_argument('--sunspots_path'   , type=str  , default='sunspots.csv'     , help='Filename of sunspots CSV in data/')
    parser.add_argument('--data_dir'        , type=str  , default='data'             , help='Directory containing input CSV files')
    parser.add_argument('--output_dir'      , type=str  , default='output'           , help='Directory to save outputs and plots')
    parser.add_argument('--seq_len'         , type=int  , default=48                 , help='Input sequence length in hours')
    parser.add_argument('--forecast_horizon', type=int  , default=24                 , help='Forecast horizon steps in hours')
    parser.add_argument('--batch_size'      , type=int  , default=64                 , help='Batch size for training and evaluation')
    parser.add_argument('--epochs'          , type=int  , default=20                 , help='Number of training epochs')
    parser.add_argument('--lr'              , type=float, default=1e-4               , help='Learning rate')
    parser.add_argument('--seed'            , type=int  , default=42                 , help='Random seed')
    parser.add_argument('--eval_only'       , action='store_true'                    , help='Skip training and evaluate using saved checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'best_dst_model.pt')

    # 1. Load preprocessed dataframe
    print("Loading and preprocessing solar wind and geomagnetic dataset...")
    df_magnet = load_and_preprocess_magnet_data(solar_wind_path = args.solar_wind_path,
                                                labels_path     = args.labels_path,
                                                sat_pos_path    = args.sat_pos_path,
                                                sunspots_path   = args.sunspots_path,
                                                data_dir        = args.data_dir)

    # 2. Chronological Train / Val / Test Split (70% / 10% / 20%)
    train_end = int(len(df_magnet) * 0.7)
    val_end   = int(len(df_magnet) * 0.8)
    train_df  = df_magnet.iloc[:train_end].copy()
    val_df    = df_magnet.iloc[train_end:val_end].copy()
    test_df   = df_magnet.iloc[val_end:].copy()

    # 3. Instantiate Datasets
    train_dataset = MagNetDataset(train_df, seq_len=args.seq_len, forecast_horizon=args.forecast_horizon, scaler=None                , fit_scaler=True)
    val_dataset   = MagNetDataset(val_df  , seq_len=args.seq_len, forecast_horizon=args.forecast_horizon, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset  = MagNetDataset(test_df , seq_len=args.seq_len, forecast_horizon=args.forecast_horizon, scaler=train_dataset.scaler, fit_scaler=False)

    # 4. Weighted Sampler for Storm Events (Cleaned)
    y_tensor          = train_dataset.y
    train_dst_targets = (y_tensor[:, 0] if y_tensor.ndim > 1 else y_tensor).numpy()
    train_dst_targets = np.nan_to_num(train_dst_targets, nan=0.0)
    sample_weights    = np.where(train_dst_targets < -30.0, 10.0, 1.0)
    sample_weights    = np.nan_to_num(sample_weights, nan=1.0)
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader   = DataLoader(val_dataset  , batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset , batch_size=args.batch_size, shuffle=False)
    print(f"DataLoaders initialized with {len(train_dataset.feature_cols)} input features.")

    # 6. Model Initialization
    num_features = len(train_dataset.feature_cols)
    model = PhysicsTCNLSTM(input_dim        = num_features,
                           num_channels     = [32, 64, 128],
                           lstm_hidden      = 64,
                           forecast_horizon = args.forecast_horizon,
                           dropout          = 0.4
                            ).to(device)

    # 7. Loss, Optimizer & Scheduler
    criterion = PhysicsGuidedDstLoss(alpha=8.0, lambda_phys=0.05, threshold=-30.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 8. Training Loop
    if not args.eval_only:
        print("\nStarting Model Training...")
        best_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
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
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{best_marker}")
        print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")

    # 9. Model Evaluation & Plot Generation
    print("\nRunning Evaluation on Test Dataset...")
    evaluate_no_history_models(model, 
                               test_loader, 
                               test_df         = test_df,
                               checkpoint_path = checkpoint_path, 
                               output_dir      = args.output_dir,
                               device          = device)

if __name__ == '__main__':
    main()