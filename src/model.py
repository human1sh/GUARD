import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """
    Chomps off trailing padding to preserve causal temporal processing.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated Temporal Convolutional Block with Residual Highway.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PhysicsTCNLSTM(nn.Module):
    """
    Multi-Horizon Hybrid TCN-LSTM Model for Geomagnetic Dst Index Forecasting.
    Supports point estimation or multi-quantile output (p10, p50, p90).
    """
    def __init__(
        self, 
        input_dim, 
        num_channels=[16, 32, 64, 128], 
        lstm_hidden=128, 
        forecast_horizon=24, 
        quantiles=None, 
        dropout=0.2
    ):
        super(PhysicsTCNLSTM, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles) if quantiles else 1
        
        tcn_layers = []
        in_ch = input_dim
        kernel_size = 3
        
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            tcn_layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding, dropout=dropout)
            )
            in_ch = out_ch
            
        self.tcn = nn.Sequential(*tcn_layers)
        self.lstm = nn.LSTM(input_size=num_channels[-1], hidden_size=lstm_hidden, batch_first=True)
        
        # Linear Head projecting to (forecast_horizon * num_quantiles)
        output_dim = self.forecast_horizon * self.num_quantiles
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x_tcn = x.transpose(1, 2)
        tcn_out = self.tcn(x_tcn)
        tcn_out = tcn_out.transpose(1, 2)
        
        lstm_out, _ = self.lstm(tcn_out)
        out = self.fc(lstm_out[:, -1, :])
        
        if self.quantiles:
            out = out.view(-1, self.forecast_horizon, self.num_quantiles)
        elif self.forecast_horizon == 1:
            out = out.squeeze(-1)  # Ensure shape [batch_size] instead of [batch_size, 1]
            
        return out