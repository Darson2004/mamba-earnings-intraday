"""
Advanced Financial Normalization for MambaStock

Implements feature-aware normalization for the 13 financial features:
[open, high, low, close, volume, pe, turnover_rate, total_share, float_share, 
 trade_time_min, pct_chg, turnover_rate_mask, pe_nan_mask]

This normalization is designed specifically for financial time series data
and handles the different statistical properties of each feature type.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

class AdvancedFinancialNormalizer:
    """
    Advanced normalizer that handles different feature types with appropriate transformations:
    
    - Price features (OHLC): Log-relative to window start
    - Volume: De-seasonalized by minute-of-day + log1p
    - Percentage change: Log1p + clipping + window z-score
    - PE ratio: Winsorization + log1p + cross-sectional robust scaling
    - Turnover rate: Context-dependent (logit vs de-seasonalized)
    - Share counts: Log1p + cross-sectional scaling
    - Time: Normalized to [0,1]
    - Binary masks: Keep as {0,1}
    """
    
    def __init__(self, cross_sectional_stats: Optional[Dict[str, Any]] = None):
        """
        Initialize the normalizer with optional pre-computed cross-sectional statistics.
        
        Args:
            cross_sectional_stats: Pre-computed statistics for cross-sectional features
        """
        self.cross_sectional_stats = cross_sectional_stats or {}
        
        # Feature indices (based on dataset.py order)
        self.feature_indices = {
            'open': 0,
            'high': 1, 
            'low': 2,
            'close': 3,
            'volume': 4,
            'pe': 5,
            'turnover_rate': 6,
            'total_share': 7,
            'float_share': 8,
            'trade_time_min': 9,
            'pct_chg': 10,
            'turnover_rate_mask': 11,
            'pe_nan_mask': 12
        }
        
        # Constants
        self.eps = 1e-8
        self.pe_winsorize_bounds = [0, 100]
        self.pct_chg_clip_bounds = [-0.5, 0.5]
        self.logit_epsilon = 1e-6
        
    def _log_relative_price_transform(self, prices: torch.Tensor, close_start: torch.Tensor) -> torch.Tensor:
        """
        Transform prices to log-relative form: log(price) - log(close_at_window_start)
        
        Args:
            prices: Price tensor [batch_size, seq_len] or [batch_size, seq_len, 1]
            close_start: Close price at window start [batch_size] or [batch_size, 1]
            
        Returns:
            Log-relative prices
        """
        # Ensure positive prices
        prices = torch.clamp(prices, min=self.eps)
        close_start = torch.clamp(close_start, min=self.eps)
        
        # Handle broadcasting
        if prices.dim() == 3:
            prices = prices.squeeze(-1)
        if close_start.dim() == 1:
            close_start = close_start.unsqueeze(-1)
            
        log_prices = torch.log(prices)
        log_close_start = torch.log(close_start)
        
        return log_prices - log_close_start
    
    def _de_seasonalize_volume(self, volume: torch.Tensor, time_mins: torch.Tensor, feature_type: str = 'volume') -> torch.Tensor:
        """
        De-seasonalize volume/turnover by minute-of-day pattern using precomputed minute-of-day medians.
        
        Args:
            volume: Volume/turnover tensor [batch_size, seq_len]
            time_mins: Time in minutes [batch_size, seq_len]
            feature_type: 'volume' or 'turnover' to select appropriate medians
            
        Returns:
            De-seasonalized volume/turnover
        """
        # Select appropriate minute medians
        if feature_type == 'turnover':
            median_key = 'minute_medians_turnover'
        else:
            median_key = 'minute_medians'
            
        # Use precomputed minute-of-day medians from cross-sectional stats
        if median_key not in self.cross_sectional_stats:
            # Fallback: use mean volume as de-seasonalization factor
            print(f"⚠️  Warning: No precomputed {feature_type} minute-of-day medians available, using mean fallback")
            mean_vol = volume.mean()
            return volume / (mean_vol + self.eps)
        
        minute_medians = self.cross_sectional_stats[median_key]
        volume_adj = volume.clone()
        
        # Vectorized de-seasonalization
        for batch_idx in range(volume.shape[0]):
            for seq_idx in range(volume.shape[1]):
                minute = int(time_mins[batch_idx, seq_idx].item())
                # Clamp minute to valid range [0, 389] for trading day
                minute = max(0, min(389, minute))
                
                if minute in minute_medians:
                    median_vol = minute_medians[minute]
                    volume_adj[batch_idx, seq_idx] = volume[batch_idx, seq_idx] / (median_vol + self.eps)
                else:
                    # Fallback for missing minutes
                    volume_adj[batch_idx, seq_idx] = volume[batch_idx, seq_idx] / (volume.mean() + self.eps)
                        
        return volume_adj
    
    def _window_z_score(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization over the sequence dimension for each stock.
        
        Args:
            tensor: Input tensor [batch_size, seq_len]
            
        Returns:
            Z-score normalized tensor
        """
        # Compute mean and std over seq_len dimension for each stock
        mean_val = tensor.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std_val = tensor.std(dim=1, keepdim=True, unbiased=False)    # [batch_size, 1] - more stable for short windows
        
        # Avoid division by zero
        std_val = torch.clamp(std_val, min=self.eps)
        
        return (tensor - mean_val) / std_val
    
    def _cross_sectional_robust_scale(self, tensor: torch.Tensor, feature_name: str) -> torch.Tensor:
        """
        Apply cross-sectional robust scaling using pre-computed statistics.
        
        Args:
            tensor: Input tensor [batch_size, seq_len]
            feature_name: Name of the feature for retrieving stats
            
        Returns:
            Robust scaled tensor
        """
        if feature_name not in self.cross_sectional_stats:
            # Fallback to within-batch robust scaling
            median_val = torch.median(tensor, dim=0, keepdim=True)[0]
            q75 = torch.quantile(tensor, 0.75, dim=0, keepdim=True)
            q25 = torch.quantile(tensor, 0.25, dim=0, keepdim=True)
            iqr = q75 - q25
            iqr = torch.clamp(iqr, min=self.eps)
            return (tensor - median_val) / iqr
        
        # Use pre-computed cross-sectional statistics
        stats = self.cross_sectional_stats[feature_name]
        median_val = stats['median']
        iqr = stats['iqr']
        
        # Ensure device/dtype match the input tensor
        if isinstance(median_val, torch.Tensor):
            median_val = median_val.to(tensor.device, dtype=tensor.dtype)
        if isinstance(iqr, torch.Tensor):
            iqr = iqr.to(tensor.device, dtype=tensor.dtype)
        
        return (tensor - median_val) / iqr
    
    def _logit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply logit transformation for features that are fractions in (0,1).
        
        Args:
            tensor: Input tensor with values in (0,1)
            
        Returns:
            Logit transformed tensor
        """
        # Clip to avoid extreme values
        tensor_clipped = torch.clamp(tensor, self.logit_epsilon, 1 - self.logit_epsilon)
        return torch.log(tensor_clipped / (1 - tensor_clipped))
    
    def apply_normalization(self, input_tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply advanced financial normalization to the input tensor.
        
        Args:
            input_tensor: Input tensor [batch_size, current_seq_len, num_features]
            seq_len: Target sequence length to use for normalization window
            
        Returns:
            Normalized tensor [batch_size, seq_len, num_features]
        """
        # Extract the last seq_len window
        if input_tensor.shape[1] >= seq_len:
            window = input_tensor[:, -seq_len:, :].clone()
        else:
            # Pad if necessary (shouldn't happen in normal operation)
            pad_size = seq_len - input_tensor.shape[1]
            padding = torch.zeros(input_tensor.shape[0], pad_size, input_tensor.shape[2], 
                                device=input_tensor.device, dtype=input_tensor.dtype)
            window = torch.cat([padding, input_tensor], dim=1)
        
        batch_size, seq_length, num_features = window.shape
        normalized = window.clone()
        
        # 1. Price features (0,1,2,3): Log-relative to window start + window z-score
        close_start = window[:, 0, self.feature_indices['close']]  # [batch_size]
        
        for price_feature in ['open', 'high', 'low', 'close']:
            idx = self.feature_indices[price_feature]
            prices = window[:, :, idx]  # [batch_size, seq_len]
            
            # Log-relative transformation
            log_rel_prices = self._log_relative_price_transform(prices, close_start)
            
            # Window z-score normalization
            normalized[:, :, idx] = self._window_z_score(log_rel_prices)
        
        # 2. Volume (4): De-seasonalize + log1p + window z-score
        idx = self.feature_indices['volume']
        volume = window[:, :, idx]
        time_mins = window[:, :, self.feature_indices['trade_time_min']]
        
        # De-seasonalize volume
        volume_deseasoned = self._de_seasonalize_volume(volume, time_mins)
        
        # Log1p transformation
        volume_log1p = torch.log1p(torch.clamp(volume_deseasoned, min=0))
        
        # Window z-score
        normalized[:, :, idx] = self._window_z_score(volume_log1p)
        
        # 3. Percentage change (10): exact log-return + soft clipping + window z-score  
        idx = self.feature_indices['pct_chg']
        pct_chg = window[:, :, idx]
        
        # Exact log-return transformation (clamp to avoid log(0))
        pct_chg_safe = torch.clamp(pct_chg, min=-0.999999)  # Prevent log(0) for r = -1
        log_return = torch.log1p(pct_chg_safe)  # Exact log(1+r) for both positive and negative returns
        
        # Soft clipping using tanh (preserves extreme values better than hard clipping)
        if 'pct_chg_scale' in self.cross_sectional_stats:
            G = self.cross_sectional_stats['pct_chg_scale']
        else:
            # Fallback: use data-driven scale (2 * std)
            G = 2.0 * log_return.std()
            G = torch.clamp(G, min=0.1)  # Reasonable minimum scale
        
        log_return_capped = G * torch.tanh(log_return / G)
        
        # Window z-score
        normalized[:, :, idx] = self._window_z_score(log_return_capped)
        
        # 4. PE ratio (5): Winsorize + log1p + cross-sectional robust scaling
        idx = self.feature_indices['pe']
        pe = window[:, :, idx]
        
        # Winsorize to sane bounds
        pe_winsorized = torch.clamp(pe, self.pe_winsorize_bounds[0], self.pe_winsorize_bounds[1])
        
        # Log1p transformation
        pe_log1p = torch.log1p(torch.clamp(pe_winsorized, min=0))
        
        # Cross-sectional robust scaling
        normalized[:, :, idx] = self._cross_sectional_robust_scale(pe_log1p, 'pe')
        
        # 5. Turnover rate (6): Context-dependent transformation (per-stock decision)
        idx = self.feature_indices['turnover_rate']
        turnover = window[:, :, idx]
        
        # Check per stock if values suggest it's a fraction (0,1) or a rate (possibly >1)
        max_turnover_per_stock = turnover.max(dim=1).values  # [batch_size]
        
        normalized_turnover = torch.zeros_like(turnover)
        for stock_idx in range(turnover.shape[0]):
            stock_turnover = turnover[stock_idx, :]  # [seq_len]
            
            if max_turnover_per_stock[stock_idx] <= 1.0:
                # Treat as fraction: logit + cross-sectional robust scaling (use fraction stats)
                stock_turnover_safe = torch.clamp(stock_turnover, self.logit_epsilon, 1-self.logit_epsilon)
                turnover_logit = self._logit_transform(stock_turnover_safe)
                # Apply cross-sectional scaling using fraction-specific stats
                normalized_turnover[stock_idx, :] = self._cross_sectional_robust_scale(
                    turnover_logit.unsqueeze(0), 'turnover_rate_fraction'
                ).squeeze(0)
            else:
                # Treat as rate: de-seasonalize + log1p + cross-sectional robust scaling (use rate stats)
                stock_time_mins = time_mins[stock_idx:stock_idx+1, :]  # [1, seq_len]
                stock_turnover_batch = stock_turnover.unsqueeze(0)  # [1, seq_len]
                
                # Use turnover-specific minute medians
                turnover_deseasoned = self._de_seasonalize_volume(stock_turnover_batch, stock_time_mins, feature_type='turnover')
                turnover_log1p = torch.log1p(torch.clamp(turnover_deseasoned, min=0))
                # Apply cross-sectional scaling using rate-specific stats
                normalized_turnover[stock_idx, :] = self._cross_sectional_robust_scale(
                    turnover_log1p, 'turnover_rate_rate'
                ).squeeze(0)
        
        normalized[:, :, idx] = normalized_turnover
        
        # 6. Share counts (7,8): log1p + cross-sectional robust scaling
        for share_feature in ['total_share', 'float_share']:
            idx = self.feature_indices[share_feature]
            shares = window[:, :, idx]
            
            # Log1p transformation
            shares_log1p = torch.log1p(torch.clamp(shares, min=0))
            
            # Cross-sectional robust scaling
            normalized[:, :, idx] = self._cross_sectional_robust_scale(shares_log1p, share_feature)
        
        # 7. Time (9): Normalize to [0,1] range
        idx = self.feature_indices['trade_time_min']
        time_mins = window[:, :, idx]
        
        # Assume trading day from 0 to 390 minutes (9:30 AM to 4:00 PM)
        time_normalized = time_mins / 390.0
        normalized[:, :, idx] = torch.clamp(time_normalized, 0, 1)
        
        # 8. Binary masks (11,12): Keep as {0,1}
        for mask_feature in ['turnover_rate_mask', 'pe_nan_mask']:
            idx = self.feature_indices[mask_feature]
            # Keep masks unchanged (they should already be 0 or 1)
            normalized[:, :, idx] = window[:, :, idx]
        
        return normalized


def create_advanced_normalizer(full_dataset_data: Optional[torch.Tensor] = None, 
                             use_historical_only: bool = True) -> AdvancedFinancialNormalizer:
    """
    Create an advanced normalizer with pre-computed cross-sectional statistics.
    
    Args:
        full_dataset_data: Full dataset [n_stocks, n_timesteps, n_features] for computing stats
        use_historical_only: If True, only use first 80% of data to avoid temporal leakage
        
    Returns:
        Configured AdvancedFinancialNormalizer
    """
    cross_sectional_stats = {}
    
    if full_dataset_data is not None:
        print("Computing cross-sectional statistics for robust scaling...")
        
        # FIX: Prevent temporal leakage by using only historical data for minute-of-day medians
        if use_historical_only:
            # Use only first 80% of timesteps to avoid future information leakage
            n_timesteps = full_dataset_data.shape[1]
            historical_end = int(0.8 * n_timesteps)
            historical_data = full_dataset_data[:, :historical_end, :]
            print(f"  Using historical data only: {historical_data.shape[1]}/{n_timesteps} timesteps (80%)")
        else:
            historical_data = full_dataset_data
            print(f"  Using full dataset: {historical_data.shape[1]} timesteps")
        
        # Feature indices
        feature_indices = {
            'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4,
            'pe': 5, 'turnover_rate': 6, 'total_share': 7, 'float_share': 8,
            'trade_time_min': 9, 'pct_chg': 10, 'turnover_rate_mask': 11, 'pe_nan_mask': 12
        }
        
        # Compute minute-of-day medians for volume/turnover de-seasonalization
        print("  Computing minute-of-day medians for volume de-seasonalization...")
        volume_data = historical_data[:, :, feature_indices['volume']]  # [n_stocks, historical_timesteps]
        turnover_data = historical_data[:, :, feature_indices['turnover_rate']]  # [n_stocks, historical_timesteps]
        time_data = historical_data[:, :, feature_indices['trade_time_min']]  # [n_stocks, historical_timesteps]
        
        # Volume minute medians
        minute_medians = {}
        for minute in range(390):  # 0 to 389 minutes in trading day
            minute_mask = (time_data == minute)
            minute_volumes = volume_data[minute_mask]
            if len(minute_volumes) > 0:
                minute_medians[minute] = torch.median(minute_volumes[torch.isfinite(minute_volumes)]).item()
        
        cross_sectional_stats['minute_medians'] = minute_medians
        print(f"  Computed volume medians for {len(minute_medians)} minutes of trading day")
        
        # Turnover minute medians (separate from volume)
        minute_medians_turnover = {}
        for minute in range(390):  # 0 to 389 minutes in trading day
            minute_mask = (time_data == minute)
            minute_turnovers = turnover_data[minute_mask]
            if len(minute_turnovers) > 0:
                minute_medians_turnover[minute] = torch.median(minute_turnovers[torch.isfinite(minute_turnovers)]).item()
        
        cross_sectional_stats['minute_medians_turnover'] = minute_medians_turnover
        print(f"  Computed turnover medians for {len(minute_medians_turnover)} minutes of trading day")
        
        # Compute percentage change scale for soft clipping
        print("  Computing percentage change scale for soft clipping...")
        pct_chg_data = historical_data[:, :, feature_indices['pct_chg']].flatten()
        pct_chg_data = pct_chg_data[torch.isfinite(pct_chg_data)]
        if len(pct_chg_data) > 0:
            pct_chg_safe = torch.clamp(pct_chg_data, min=-0.999999)
            log_returns = torch.log1p(pct_chg_safe)
            
            # FIX: Use 99.9% percentile instead of 2*std to reduce compression
            # This captures more extreme movements while still capping outliers
            percentile_995 = torch.quantile(torch.abs(log_returns), 0.995)
            pct_chg_scale = torch.clamp(percentile_995, min=0.1)
            
            cross_sectional_stats['pct_chg_scale'] = pct_chg_scale.item()
            print(f"  pct_chg_scale (99.5% percentile): {cross_sectional_stats['pct_chg_scale']:.4f}")
            
            # Compare with old method for debugging
            old_scale = 2.0 * log_returns.std(unbiased=False)
            print(f"  old_scale (2*std): {old_scale:.4f}, ratio: {cross_sectional_stats['pct_chg_scale']/old_scale:.2f}x")

        # Compute robust statistics for cross-sectional features
        for feature_name in ['pe', 'total_share', 'float_share']:
            idx = feature_indices[feature_name]
            feature_data = historical_data[:, :, idx].flatten()
            
            # Remove NaN and infinite values
            feature_data = feature_data[torch.isfinite(feature_data)]
            
            if len(feature_data) > 0:
                # Apply appropriate transformation before computing stats
                if feature_name == 'pe':
                    # Winsorize + log1p
                    feature_data = torch.clamp(feature_data, 0, 100)
                    feature_data = torch.log1p(torch.clamp(feature_data, min=0))
                elif feature_name in ['total_share', 'float_share']:
                    # Log1p
                    feature_data = torch.log1p(torch.clamp(feature_data, min=0))
                
                # Compute robust statistics
                median_val = torch.median(feature_data)
                q75 = torch.quantile(feature_data, 0.75)
                q25 = torch.quantile(feature_data, 0.25)
                iqr = torch.clamp(q75 - q25, min=1e-8)
                
                cross_sectional_stats[feature_name] = {
                    'median': median_val,
                    'iqr': iqr,
                    'q25': q25,
                    'q75': q75
                }
                
                print(f"  {feature_name}: median={median_val:.4f}, IQR={iqr:.4f}")
        
        # Special handling for turnover_rate: compute separate stats for fraction vs rate
        print("  Computing separate turnover_rate statistics...")
        turnover_data = historical_data[:, :, feature_indices['turnover_rate']]  # [n_stocks, historical_timesteps]
        
        # Identify fraction vs rate stocks per stock
        fraction_data = []
        rate_data = []
        
        for stock_idx in range(turnover_data.shape[0]):
            stock_turnover = turnover_data[stock_idx, :]
            stock_turnover = stock_turnover[torch.isfinite(stock_turnover)]
            
            if len(stock_turnover) > 0:
                max_turnover = torch.max(stock_turnover)
                if max_turnover <= 1.0:
                    # Treat as fraction: logit transform
                    eps = 1e-6
                    stock_turnover_safe = torch.clamp(stock_turnover, eps, 1-eps)
                    logit_turnover = torch.log(stock_turnover_safe / (1 - stock_turnover_safe))
                    fraction_data.append(logit_turnover)
                else:
                    # Treat as rate: log1p transform
                    log1p_turnover = torch.log1p(torch.clamp(stock_turnover, min=0))
                    rate_data.append(log1p_turnover)
        
        # Compute stats for fraction stocks (logit-transformed)
        if fraction_data:
            fraction_tensor = torch.cat(fraction_data)
            fraction_median = torch.median(fraction_tensor)
            fraction_q75 = torch.quantile(fraction_tensor, 0.75)
            fraction_q25 = torch.quantile(fraction_tensor, 0.25)
            fraction_iqr = torch.clamp(fraction_q75 - fraction_q25, min=1e-8)
            
            cross_sectional_stats['turnover_rate_fraction'] = {
                'median': fraction_median,
                'iqr': fraction_iqr,
                'q25': fraction_q25,
                'q75': fraction_q75
            }
            print(f"  turnover_rate_fraction: median={fraction_median:.4f}, IQR={fraction_iqr:.4f} ({len(fraction_data)} stocks)")
        
        # Compute stats for rate stocks (log1p-transformed)
        if rate_data:
            rate_tensor = torch.cat(rate_data)
            rate_median = torch.median(rate_tensor)
            rate_q75 = torch.quantile(rate_tensor, 0.75)
            rate_q25 = torch.quantile(rate_tensor, 0.25)
            rate_iqr = torch.clamp(rate_q75 - rate_q25, min=1e-8)
            
            cross_sectional_stats['turnover_rate_rate'] = {
                'median': rate_median,
                'iqr': rate_iqr,
                'q25': rate_q25,
                'q75': rate_q75
            }
            print(f"  turnover_rate_rate: median={rate_median:.4f}, IQR={rate_iqr:.4f} ({len(rate_data)} stocks)")
    
    return AdvancedFinancialNormalizer(cross_sectional_stats)


def apply_advanced_normalization(input_tensor: torch.Tensor, seq_len: int, 
                               normalizer: AdvancedFinancialNormalizer) -> torch.Tensor:
    """
    Convenience function to apply advanced normalization.
    
    Args:
        input_tensor: Input tensor [batch_size, current_seq_len, num_features]
        seq_len: Target sequence length for normalization window
        normalizer: Pre-configured normalizer
        
    Returns:
        Normalized tensor [batch_size, seq_len, num_features]
    """
    return normalizer.apply_normalization(input_tensor, seq_len)