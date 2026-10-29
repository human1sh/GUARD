# 🌌 MagNet: Physics-Informed Geomagnetic Storm Forecasting

> **Predicting Earth's $Dst$ geomagnetic index using hybrid TCN-LSTM networks guided by non-linear magnetospheric physics, pre-aligned L1 solar wind drivers, autoregressive state grounding, and asymmetric storm loss functions.**

---

## 💡 Overview & Physical Motivation

Solar wind—streams of charged plasma and magnetic fields ejected by the Sun—constantly interacts with Earth's magnetosphere. Severe geomagnetic storms induced by Coronal Mass Ejections (CMEs) and Coronal Hole High-Speed Streams can trigger ground-induced currents (GICs) that damage electrical grids, disrupt GPS navigation, radio communications, and cause satellite orbital decay.

This repository implements **MagNet**, an end-to-end Physics-Informed Scientific Machine Learning (SciML) pipeline designed to forecast the **$Dst$ (Disturbance Storm Time) index**:
- **$Dst \ge -20\text{ nT}$**: Quiet-time geomagnetic baseline state.
- **$Dst < -30\text{ nT}$**: Moderate geomagnetic storm threshold.
- **$Dst < -50\text{ nT}$**: Severe geomagnetic storm depression.

```
L1 Solar Wind Telemetry (ACE/DSCOVR) ──► Ballistic Time-Alignment ──► Physical Coupling (Akasofu/Newell/Burton)
                                                                                  │
                                                                                  ▼
Predicted Dst (t+h) ◄── Asymmetric Loss ◄── LSTM (Ring Current Decay) ◄── Dilated TCN (Fast Triggers)
```

### Why Combine Physics with Deep Learning?

1. **Ballistic Driver Pre-Alignment**: Solar wind plasma measured at Lagrange Point L1 (~1.5 million km upstream) takes 30–60 minutes to reach Earth. We pre-align driver fields ($B_z, v_{sw}, P_{dyn}$) by calculated ballistic delay times ($\tau_{delay} = X_{L1} / v_{sw}$) to enforce physical causality within temporal sequence windows.
2. **Domain-Specific Feature Engineering**: Beyond raw IMF vector components ($B_y, B_z$), the network processes 16 engineered channels including:
   - **Interplanetary Electric Field ($IEF = v \cdot B_z$)**: Direct convection driver.
   - **Dynamic Pressure ($P_{dyn} = \rho v^2$)**: Magnetospheric boundary compression.
   - **Newell Coupling Term ($d\Phi_{MP}/dt$)**: Non-linear energy transfer rate across the magnetopause.
   - **Akasofu $\epsilon$ Energy Input**: Poynting flux proxy for magnetospheric power input.
   - **Burton-McPherron Ring Current Kinetics**: Differential kinetic proxy for ring current energy accumulation and exponential relaxation.
3. **Autoregressive State Grounding**: Incorporating lagged target features ($Dst_{t-1}, Dst_{t-2}$) grounds predictions on Earth's real-time magnetospheric energy state, eliminating baseline drift and anchor floating.
4. **Asymmetric Geomagnetic Loss**: Standard MSE penalizes quiet-time noise equally with storm dips. Our custom `AsymmetricGeomagneticLoss` applies a dynamic weight multiplier ($\alpha$) when the network underpredicts storm depressions ($Dst < -30\text{ nT}$), driving 100% detection reliability for severe space weather events.

---

## 📈 Model Performance & Benchmarks

Evaluation on the test dataset over 1-hour ahead forecasts ($1\text{h}$-ahead):

| Performance Metric | Baseline Model (Pure Solar Wind) | MagNet (Physics + Lags + Cosine Annealing) | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Best Validation Loss** | `6.3440` | **`3.1543`** | **~50.3% Error Reduction** |
| **Variance Explained ($R^2$)** | `0.606` | **`0.845`** | **+39.4% Variance Explained** |
| **Pearson Correlation ($r$)** | `0.789` | **`0.920`** | Near-Linear Phase Tracking |
| **Overall RMSE** | `6.937 nT` | **`4.338 nT`** | **~37.5% Lower Global Error** |
| **Overall MAE** | `5.235 nT` | **`3.109 nT`** | **~40.6% Lower Absolute Error** |
| **Storm RMSE ($Dst < -30\text{ nT}$)** | `13.342 nT` | **`10.270 nT`** | **~23.0% Error Reduction** |
| **Severe Storm RMSE ($Dst < -50\text{ nT}$)** | `20.925 nT` | **`17.336 nT`** | **~17.2% Error Reduction** |

> **Operational Threat Alarm Reliability**: 100% detection rate for actual severe storms crossing the $-30\text{ nT}$ moderate threshold (zero missed alarms).

---

## 📂 Dataset Source & Preparation

This project uses the benchmark **NASA and NOAA Satellites Solar-Wind Dataset** hosted on Kaggle:
- **Kaggle Dataset**: [NASA and NOAA Satellites Solar-Wind Dataset by Arashnic](https://www.kaggle.com/datasets/arashnic/soalr-wind)

### Required File Structure
Download and place the dataset CSV files inside the `data/` directory:
- `solar_wind.csv`: ACE/DSCOVR IMF and plasma measurements ($B_x, B_y, B_z, v_{sw}$, density, temperature).
- `labels.csv`: Hourly ground truth geomagnetic $Dst$ index values.
- `satellite_pos.csv`: Ephemeris trajectories for L1 observation satellites.
- `sunspots.csv`: Solar cycle activity metrics.

---

## 📁 Repository Structure

```text
magnet_dst_prediction/
│
├── main.py               # Central CLI execution entry point
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules for data, models, and outputs
│
├── src/                  # Core modular package
│   ├── __init__.py       # Package initializer
│   ├── dataset.py        # Feature engineering, L1 alignment, autoregressive lags, PyTorch DataLoader
│   ├── model.py          # Temporal Convolutional Network (TCN) + LSTM architecture
│   ├── loss.py           # Custom AsymmetricGeomagneticLoss implementation
│   ├── train.py          # Training loop with Cosine Annealing scheduler and gradient clipping
│   └── evaluate.py       # Metrics calculator & publication-ready diagnostic plot generator
│
├── data/                 # Raw/Preprocessed CSV datasets (Kaggle dataset target)
└── output/               # Checkpoints and diagnostic visualizations
    ├── best_dst_model.pt
    ├── dst_timeseries_prediction.png
    ├── dst_scatter_correlation.png
    └── dst_residual_distribution.png
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/magnet_dst_prediction.git](https://github.com/your-username/magnet_dst_prediction.git)
   cd magnet_dst_prediction
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Populate Dataset:**
   Download the [Kaggle Solar Wind Dataset](https://www.kaggle.com/datasets/arashnic/soalr-wind) and extract all CSV files into the `data/` folder.

---

## 🚀 Usage

### 1. Training & Evaluation
To train the hybrid TCN-LSTM model from scratch using Cosine Annealing learning rate scheduling and pre-aligned physics features:

```bash
python main.py --seq_len 72 --forecast_horizon 1 --lr 0.0001 --epochs 20 --batch_size 64
```

### 2. Evaluation Mode Only
To load the best trained checkpoint (`output/best_dst_model.pt`) and re-generate diagnostic plots and metrics without retraining:

```bash
python main.py --eval_only --forecast_horizon 1 --seq_len 72
```

---

## 📊 Outputs & Visualizations

All diagnostic outputs are saved to `output/`:
- **`best_dst_model.pt`**: Serialized PyTorch state dict of the top-performing checkpoint.
- **`dst_timeseries_prediction.png`**: Full test sequence plot and storm main-phase zoom-in cutouts.
- **`dst_scatter_correlation.png`**: Observed vs. Predicted $Dst$ scatter plot with $1:1$ reference line and storm threshold overlays.
- **`dst_residual_distribution.png`**: Residual error density histogram ($y_{\text{true}} - y_{\text{pred}}$) demonstrating symmetric Gaussian error distribution centered at $0\text{ nT}$.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.