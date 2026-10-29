import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.signal import find_peaks

def evaluate_no_history_models(
    model, 
    test_loader, 
    test_df=None,
    checkpoint_path=os.path.join('output', 'best_dst_model.pt'), 
    output_dir='output',
    device='cpu', 
    save_plots=True
):
    """
    Evaluates trained multi-horizon model on test dataset and generates 
    diagnostic plots for full time-series and intense storm events.
    """
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint from '{checkpoint_path}'.")
    except FileNotFoundError:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Evaluating current model weights.")

    model.eval()
    tcn_preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            tcn_preds.append(preds)
            targets.append(y_batch.numpy())

    # Flatten and concatenate batches safely, handling partial final batches and 1D/2D outputs
    tcn_preds_flat = [p.squeeze() for p in tcn_preds]
    targets_flat = [t.squeeze() for t in targets]
    y_pred_all = np.concatenate(tcn_preds_flat, axis=0)
    y_true_all = np.concatenate(targets_flat, axis=0)

    # ==========================================
    # STEP 2: MULTI-HORIZON PERFORMANCE BREAKDOWN
    # ==========================================
    print("\n================ MULTI-HORIZON PERFORMANCE ==================")
    horizon_steps = [0, 2, 5, 11, 23]  # Corresponds to t+1h, t+3h, t+6h, t+12h, t+24h
    for h in horizon_steps:
        if y_pred_all.ndim > 1 and h < y_pred_all.shape[1]:
            yp_h = y_pred_all[:, h]
            yt_h = y_true_all[:, h]
            r2_h = r2_score(yt_h, yp_h)
            rmse_h = np.sqrt(mean_squared_error(yt_h, yp_h))
            print(f"  - Horizon t+{(h+1):02d}h  |  R^2: {r2_h:.3f}  |  RMSE: {rmse_h:.3f} nT")
    print("=============================================================\n")

    # Evaluate 1-hour ahead forecast across the test set (Primary Baseline)
    y_pred_1h = y_pred_all[:, 0] if y_pred_all.ndim > 1 else y_pred_all.squeeze()
    y_true_1h = y_true_all[:, 0] if y_true_all.ndim > 1 else y_true_all.squeeze()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true_1h, y_pred_1h))
    mae = mean_absolute_error(y_true_1h, y_pred_1h)
    r2 = r2_score(y_true_1h, y_pred_1h)
    corr, _ = pearsonr(y_true_1h, y_pred_1h)

    storm_mask_30 = y_true_1h < -30.0
    storm_mask_50 = y_true_1h < -50.0

    rmse_30 = np.sqrt(mean_squared_error(y_true_1h[storm_mask_30], y_pred_1h[storm_mask_30])) if np.sum(storm_mask_30) > 0 else np.nan
    rmse_50 = np.sqrt(mean_squared_error(y_true_1h[storm_mask_50], y_pred_1h[storm_mask_50])) if np.sum(storm_mask_50) > 0 else np.nan

    print("\n================ MAGNET MODEL EVALUATION ================")
    print(f"Overall Dataset Metrics (1h-ahead):")
    print(f"  - RMSE:               {rmse:.3f} nT")
    print(f"  - MAE:                {mae:.3f} nT")
    print(f"  - R^2 Score:          {r2:.3f}")
    print(f"  - Pearson Corr (r):   {corr:.3f}")
    print("---------------------------------------------------------")
    print(f"Geomagnetic Storm Metrics:")
    print(f"  - Storm RMSE (Dst < -30 nT): {rmse_30:.3f} nT  (n={np.sum(storm_mask_30)})")
    print(f"  - Storm RMSE (Dst < -50 nT): {rmse_50:.3f} nT  (n={np.sum(storm_mask_50)})")
    print("=========================================================\n")

    if save_plots:
        plot_evaluation_figures(y_true_1h, y_pred_1h, test_df=test_df, output_dir=output_dir)

    return y_true_1h, y_pred_1h


def plot_evaluation_figures(y_true, y_pred, test_df=None, output_dir='output', storm_threshold=-50.0):
    """
    Generates and saves full time-series, multi-storm event zooms, scatter, and residual plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    time_labels = None
    if test_df is not None:
        if 'timedelta' in test_df.columns:
            time_labels = test_df['timedelta'].values
        elif 'period' in test_df.columns:
            time_labels = test_df['period'].values

    # 1. Full Test Set Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label='Observed $Dst$', color='black', alpha=0.8, linewidth=1.0)
    ax.plot(y_pred, label='Predicted $Dst$ (Physics TCN-LSTM)', color='crimson', alpha=0.8, linewidth=1.0)
    ax.axhline(-30, color='orange', linestyle='--', linewidth=0.8, label='Moderate Storm (-30 nT)')
    ax.axhline(-50, color='red', linestyle='--', linewidth=0.8, label='Intense Storm (-50 nT)')
    ax.set_title('Geomagnetic $Dst$ Index Prediction: Full Test Series', fontsize=12, fontweight='bold')
    ax.set_ylabel('$Dst$ (nT)')
    ax.set_xlabel('Time Step (Hours)')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dst_timeseries_prediction.png'), dpi=300)
    plt.close()

    # 2. Intense Storm Event Zoom Grid Labeled with Occurrence Time
    peaks, _ = find_peaks(-y_true, height=-storm_threshold, distance=48)

    if len(peaks) > 0:
        num_storms = len(peaks)
        fig, axes = plt.subplots(num_storms, 1, figsize=(14, 3.5 * num_storms), sharey=True)
        if num_storms == 1:
            axes = [axes]

        for i, idx in enumerate(peaks):
            start = max(0, idx - 36)
            end = min(len(y_true), idx + 36)

            time_str = f"Time: {time_labels[idx]}" if time_labels is not None and idx < len(time_labels) else f"Step: {idx}h"
            title = f"Storm Event #{i+1} | {time_str} | Peak Obs Dst: {y_true[idx]:.1f} nT (Pred: {y_pred[idx]:.1f} nT)"

            axes[i].plot(range(start, end), y_true[start:end], label='Observed $Dst$', color='black', linewidth=1.5)
            axes[i].plot(range(start, end), y_pred[start:end], label='Predicted $Dst$', color='crimson', linewidth=1.5)
            axes[i].axhline(-30, color='orange', linestyle='--', linewidth=1.2, label='Moderate Storm (-30 nT)')
            axes[i].axhline(-50, color='red'   , linestyle='--', linewidth=1.2, label='Intense    Storm (-50 nT)')
            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].set_ylabel('$Dst$ (nT)')
            axes[i].legend(loc='lower left', frameon=True)

        axes[-1].set_xlabel('Time Step (Hours)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dst_intense_storms.png'), dpi=300)
        plt.close()

    # 3. Scatter Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, color='navy', s=10, edgecolors='none')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1 Line', linewidth=1.5)
    plt.axvline(-30, color='orange', linestyle=':', label='Storm Threshold (-30 nT)')
    plt.axhline(-30, color='orange', linestyle=':')
    plt.title('Observed vs Predicted $Dst$ Index', fontsize=12, fontweight='bold')
    plt.xlabel('Observed $Dst$ (nT)')
    plt.ylabel('Predicted $Dst$ (nT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dst_scatter_correlation.png'), dpi=300)
    plt.close()

    # 4. Residual Distribution
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, color='teal', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title('Prediction Residual Error Distribution ($y_{true} - y_{pred}$)', fontsize=12, fontweight='bold')
    plt.xlabel('Error (nT)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dst_residual_distribution.png'), dpi=300)
    plt.close()

    print(f"Evaluation figures saved to directory: '{output_dir}/'")