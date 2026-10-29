# 🌌 GUARD: Physics-Informed Geomagnetic Storm Forecasting

> **Predicting Earth's $Dst$ geomagnetic index using hybrid TCN-LSTM networks guided by non-linear magnetospheric physics, pre-aligned L1 solar wind drivers, autoregressive state grounding, and asymmetric storm loss functions.**

---

## 💡 Overview & Physical Motivation

Solar wind—streams of charged plasma and magnetic fields ejected by the Sun—constantly interacts with Earth's magnetosphere. Severe geomagnetic storms induced by Coronal Mass Ejections (CMEs) and Coronal Hole High-Speed Streams can trigger ground-induced currents (GICs) that damage electrical grids, disrupt GPS navigation, radio communications, and cause satellite orbital decay.

This repository implements **GUARD** (**G**eomagnetic **U**pstream **A**lert & **R**isk **D**etector), an end-to-end Physics-Informed Scientific Machine Learning (SciML) pipeline designed to forecast the **$Dst$ (Disturbance Storm Time) index**:
- **$Dst \ge -20\text{ nT}$**: Quiet-time geomagnetic baseline state.
- **$Dst < -30\text{ nT}$**: Moderate geomagnetic storm threshold.
- **$Dst < -50\text{ nT}$**: Severe geomagnetic storm depression.

```text
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

| Performance Metric | Baseline Model (Pure Solar Wind) | GUARD (Physics + Lags + Cosine Annealing) | Improvement / Delta |
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

## 📥 Data Setup

The raw dataset (~900 MB) is excluded from version control to keep the repository lightweight. Follow these steps to set up the data locally before running the code:

### Option 1: Manual Download (Recommended)
1. Download the dataset from Kaggle: **[NASA and NOAA Satellites Solar-Wind Dataset](https://www.kaggle.com/datasets/arashnic/soalr-wind)**.
2. Extract the contents into a folder named `data/` in the root directory of this repository:
   ```text
   guard_dst_prediction/
   ├── data/
   │   ├── solar_wind.csv
   │   ├── labels.csv
   │   ├── satellite_pos.csv
   │   └── sunspots.csv
   ├── src/
   ├── main.py
   └── README.md
   ```

### Option 2: Via Kaggle CLI
If you have the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) set up on your system, run:

```bash
mkdir -p data
kaggle datasets download -d arashnic/soalr-wind --unzip -p data/
```

---

## 📁 Repository Structure

```text
guard_dst_prediction/
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
   git clone [https://github.com/your-username/guard_dst_prediction.git](https://github.com/your-username/guard_dst_prediction.git)
   cd guard_dst_prediction
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