import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

def load_and_preprocess_magnet_data(
    solar_wind_path='solar_wind.csv',
    labels_path='labels.csv',
    sat_pos_path='satellite_pos.csv',
    sunspots_path='sunspots.csv',
    data_dir='data',
    tau_hours=8.0
):
    df_sw = pd.read_csv(os.path.join(data_dir, solar_wind_path))
    df_labels = pd.read_csv(os.path.join(data_dir, labels_path))
    df_sat = pd.read_csv(os.path.join(data_dir, sat_pos_path))
    df_sun = pd.read_csv(os.path.join(data_dir, sunspots_path))

    # Clean missing sentinels
    df_sw = df_sw.replace([-9999.0, -999.9, 9999.0, 999.9], np.nan)
    if 'dst' in df_labels.columns:
        df_labels['dst'] = df_labels['dst'].replace([-9999.0, -999.9, 9999.0], np.nan)

    # Merges
    df = pd.merge(df_sw, df_labels, on=['period', 'timedelta'], how='inner')
    if 'period' in df_sat.columns and 'timedelta' in df_sat.columns:
        df = pd.merge(df, df_sat, on=['period', 'timedelta'], how='left')
    if 'period' in df_sun.columns and 'timedelta' in df_sun.columns:
        df = pd.merge(df, df_sun, on=['period', 'timedelta'], how='left')

    bz_col = 'bz_gsm' if 'bz_gsm' in df.columns else 'bz_gse'
    by_col = 'by_gsm' if 'by_gsm' in df.columns else 'by_gse'
    bx_col = 'bx_gsm' if 'bx_gsm' in df.columns else 'bx_gse'

    # Clean and interpolate raw numerical columns first
    raw_cols = [bz_col, by_col, bx_col, 'bt', 'speed', 'density', 'temperature', 'dst']
    for c in raw_cols:
        if c in df.columns:
            df[c] = df[c].interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0.0)

    # 1. Physics Features (Bs, IEF, P_dyn)
    df['Bs'] = np.maximum(0.0, -df[bz_col])
    df['IEF'] = df['speed'] * df['Bs'] * 1e-3
    
    density_safe = np.clip(df['density'], 0.01, 200.0)
    speed_safe = np.clip(df['speed'], 100.0, 2000.0)
    df['P_dyn'] = (density_safe * 1e6) * (1.6726e-27) * ((speed_safe * 1e3) ** 2) * 1e9

    # 2. Coupling Functions
    eps = 1e-6
    B_T = np.sqrt(np.maximum(eps, df[by_col]**2 + df[bz_col]**2))
    B_total = np.sqrt(np.maximum(eps, df[bx_col]**2 + df[by_col]**2 + df[bz_col]**2))
    theta_c = np.arctan2(df[by_col], df[bz_col])
    
    sin_term = np.abs(np.sin(theta_c / 2.0)) ** (8.0 / 3.0)
    df['newell_coupling'] = (speed_safe ** (4.0 / 3.0)) * (B_T ** (2.0 / 3.0)) * sin_term

    l0_m = 7.0 * 6371e3
    sin_akasofu = np.sin(theta_c / 2.0) ** 4
    df['akasofu_epsilon'] = 1e7 * (speed_safe * 1e3) * ((B_total * 1e-9) ** 2) * sin_akasofu * (l0_m ** 2)

    # 3. Ring Current Decay Proxy
    ief_vals = df['IEF'].values
    dst_vals = df['dst'].values if 'dst' in df.columns else np.zeros(len(df))
    e_proxy = np.zeros(len(df))
    for t in range(1, len(df)):
        current_tau = 3.5 if ief_vals[t] > 3.0 or dst_vals[t-1] < -50.0 else 12.0
        decay = np.exp(-1.0 / current_tau)
        e_proxy[t] = e_proxy[t-1] * decay + ief_vals[t]
    df['ring_current_energy'] = e_proxy

    # 4. Propagation Delay
    if 'gse_x_dscovr' in df.columns:
        x_pos_km = df['gse_x_dscovr'].fillna(df['gse_x_ace'] if 'gse_x_ace' in df.columns else 1.5e6)
    elif 'gse_x_ace' in df.columns:
        x_pos_km = df['gse_x_ace']
    else:
        x_pos_km = 1.5e6
    df['propagation_delay_min'] = (x_pos_km.fillna(1.5e6) / speed_safe) / 60.0

    # =========================================================================
    # OPTION 2: PRE-ALIGN SOLAR WIND TELEMETRY BY BALLISTIC L1-TO-EARTH LAG
    # =========================================================================
    # Compute mean travel delay in hourly steps (typically ~1 step / 60 mins)
    shift_steps = max(1, int(np.round(df['propagation_delay_min'].mean() / 60.0)))
    
    driver_cols = [
        bz_col, by_col, 'bt', 'speed', 'density', 'temperature', 
        'Bs', 'IEF', 'P_dyn', 'newell_coupling', 'akasofu_epsilon'
    ]
    
    for col in driver_cols:
        if col in df.columns:
            if 'period' in df.columns:
                df[col] = df.groupby('period')[col].shift(shift_steps)
            else:
                df[col] = df[col].shift(shift_steps)
            df[col] = df[col].bfill().fillna(0.0)

    # 5. Autoregressive Dst Lag Features (Grouped by observation period)
    if 'period' in df.columns:
        df['dst_lag1'] = df.groupby('period')['dst'].shift(1)
        df['dst_lag2'] = df.groupby('period')['dst'].shift(2)
    else:
        df['dst_lag1'] = df['dst'].shift(1)
        df['dst_lag2'] = df['dst'].shift(2)

    df['dst_lag1'] = df['dst_lag1'].bfill().fillna(0.0)
    df['dst_lag2'] = df['dst_lag2'].bfill().fillna(0.0)

    # Feature List
    feature_cols = [
        bz_col, by_col, 'bt', 'speed', 'density', 'temperature', 
        'Bs', 'IEF', 'P_dyn', 'newell_coupling', 'akasofu_epsilon', 
        'ring_current_energy', 'propagation_delay_min', 'dst_lag1', 'dst_lag2', 'dst'
    ]
    if 'smoothed_ssn' in df.columns:
        df['smoothed_ssn'] = df['smoothed_ssn'].fillna(0.0)
        feature_cols.append('smoothed_ssn')

    df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='both').bfill().ffill().fillna(0.0)

    return df

class MagNetDataset(Dataset):
    """
    PyTorch Dataset for multi-horizon Dst sequence prediction.
    """
    def __init__(self, df, seq_len=48, forecast_horizon=24, scaler=None, fit_scaler=False):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        bz_col = 'bz_gsm' if 'bz_gsm' in df.columns else 'bz_gse'
        by_col = 'by_gsm' if 'by_gsm' in df.columns else 'by_gse'
        
        self.feature_cols = [
            bz_col, by_col, 'bt', 'speed', 'density', 'temperature', 
            'Bs', 'IEF', 'P_dyn', 'newell_coupling', 'akasofu_epsilon', 
            'ring_current_energy', 'propagation_delay_min'
        ]
        if 'smoothed_ssn' in df.columns:
            self.feature_cols.append('smoothed_ssn')

        for lag_col in ['dst_lag1', 'dst_lag2']:
            if lag_col in df.columns:
                self.feature_cols.append(lag_col)

        self.target_col = ['dst']
        feature_data = df[self.feature_cols].values

        if scaler is not None:
            self.scaler = scaler
            if fit_scaler:
                self.scaler.fit(feature_data)
            scaled_features = self.scaler.transform(feature_data)
        else:
            self.scaler = RobustScaler()
            scaled_features = self.scaler.fit_transform(feature_data)

        # Replace any residual NaNs post-scaling
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(df[self.target_col].values, nan=0.0, posinf=0.0, neginf=0.0)

        X_list, y_list = [], []
        total_steps = len(df) - seq_len - forecast_horizon + 1
        for i in range(total_steps):
            X_list.append(scaled_features[i : i + seq_len])
            y_list.append(targets[i + seq_len : i + seq_len + forecast_horizon].squeeze())

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]