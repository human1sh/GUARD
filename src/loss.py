import torch
import torch.nn as nn

class AsymmetricGeomagneticLoss(nn.Module):
    """
    Asymmetric Loss Function targeting severe Dst storm depressions.
    Applies weight factor alpha to penalties when underpredicting storm dips.
    """
    def __init__(self, alpha=6.0, threshold=-30.0, use_smooth_l1=True):
        super(AsymmetricGeomagneticLoss, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        storm_underpredict_mask = (y_true < self.threshold) & (error > 0)
        
        if self.use_smooth_l1:
            base_loss = self.smooth_l1(y_pred, y_true)
        else:
            base_loss = error ** 2

        weighted_loss = torch.where(storm_underpredict_mask, self.alpha * base_loss, base_loss)
        return torch.mean(weighted_loss)


class PhysicsGuidedDstLoss(nn.Module):
    """
    PINN Loss with Horizon Decay and robust dimension alignment for 1D/2D targets.
    """
    def __init__(self, alpha=4.0, lambda_phys=0.05, threshold=-30.0, horizon_decay=0.92):
        super(PhysicsGuidedDstLoss, self).__init__()
        self.alpha = alpha
        self.lambda_phys = lambda_phys
        self.threshold = threshold
        self.horizon_decay = horizon_decay
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, y_pred, y_true):
        # Align dimensions if y_pred has trailing singleton dimension (e.g., [64, 1] vs [64])
        if y_pred.ndim > y_true.ndim:
            y_pred = y_pred.squeeze(-1)
        elif y_true.ndim > y_pred.ndim:
            y_true = y_true.squeeze(-1)

        # 1. Asymmetric Data Error Loss
        error = y_true - y_pred
        storm_mask = (y_true < self.threshold) & (error > 0)
        base_loss = self.smooth_l1(y_pred, y_true)
        data_loss = torch.where(storm_mask, self.alpha * base_loss, base_loss)

        # 2. Apply Exponential Horizon Weighting if multi-horizon
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            forecast_horizon = y_pred.shape[1]
            weights = torch.tensor([self.horizon_decay ** h for h in range(forecast_horizon)], device=y_pred.device)
            data_loss = (data_loss * weights).mean()
        else:
            data_loss = data_loss.mean()

        # 3. Temporal Rate Regularization (dDst/dt)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            d_pred = y_pred[:, 1:] - y_pred[:, :-1]
            d_true = y_true[:, 1:] - y_true[:, :-1]
            phys_loss = torch.mean((d_pred - d_true) ** 2)
        else:
            phys_loss = 0.0

        return data_loss + (self.lambda_phys * phys_loss)


class QuantileLoss(nn.Module):
    """
    Pinball Loss for Multi-Quantile Probabilistic Forecasting (e.g. p10, p50, p90).
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, forecast_horizon, num_quantiles)
        # y_true: (batch_size, forecast_horizon)
        if y_true.ndim == 2:
            y_true = y_true.unsqueeze(-1)
            
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, :, i:i+1]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
            
        return sum(losses)