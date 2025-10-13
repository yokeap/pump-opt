
# ==============================================================================
# FILE: src/proxy_functions_filtered.py
# Proxy Functions with Noise Filtering for Robust BEP Detection
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, List, Tuple

class ProxyFunction(ABC):
    """Abstract base class for efficiency proxy functions"""
    
    def __init__(self, filter_type: str = 'none', filter_params: dict = None):
        """
        Args:
            filter_type: 'none', 'savgol', 'gaussian', 'median', 'moving_average'
            filter_params: Dictionary of filter-specific parameters
        """
        self.filter_type = filter_type
        self.filter_params = filter_params or {}
        
    @abstractmethod
    def calculate(self, measurement) -> float:
        """Calculate efficiency proxy from measurement"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get proxy function name"""
        pass
    
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply selected filter to data array"""
        if self.filter_type == 'none' or len(data) < 5:
            return data
        
        elif self.filter_type == 'savgol':
            # Savitzky-Golay filter (polynomial smoothing)
            window = self.filter_params.get('window_length', 7)
            polyorder = self.filter_params.get('polyorder', 2)
            
            # Ensure window is odd and not larger than data
            window = min(window, len(data))
            if window % 2 == 0:
                window -= 1
            window = max(5, window)  # Minimum window size
            polyorder = min(polyorder, window - 1)
            
            return savgol_filter(data, window, polyorder)
        
        elif self.filter_type == 'gaussian':
            # Gaussian smoothing
            sigma = self.filter_params.get('sigma', 1.5)
            return gaussian_filter1d(data, sigma)
        
        elif self.filter_type == 'median':
            # Median filter (good for outliers)
            kernel_size = self.filter_params.get('kernel_size', 5)
            kernel_size = min(kernel_size, len(data))
            if kernel_size % 2 == 0:
                kernel_size -= 1
            kernel_size = max(3, kernel_size)
            return medfilt(data, kernel_size=kernel_size)
        
        elif self.filter_type == 'moving_average':
            # Simple moving average
            window = self.filter_params.get('window', 5)
            window = min(window, len(data))
            
            if window < 2:
                return data
            
            # Pad data at edges to maintain size
            pad_width = window // 2
            padded = np.pad(data, pad_width, mode='edge')
            
            # Convolve with uniform window
            weights = np.ones(window) / window
            smoothed = np.convolve(padded, weights, mode='valid')
            
            return smoothed
        
        else:
            return data
    
    def calculate_batch(self, measurements: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate proxy for batch of measurements and apply filtering
        
        Returns:
            raw_values: Unfiltered proxy values
            filtered_values: Filtered proxy values
        """
        raw_values = np.array([self.calculate(m) for m in measurements])
        filtered_values = self.apply_filter(raw_values)
        
        return raw_values, filtered_values


class firstOrderProxy(ProxyFunction):
    """Linear Q/P proxy with optional filtering"""
    
    def __init__(self, rated_flow: float = 100.0, 
                 filter_type: str = 'savgol',
                 filter_params: dict = None):
        """
        Args:
            rated_flow: Rated flow for normalization
            filter_type: 'none', 'savgol', 'gaussian', 'median', 'moving_average'
            filter_params: Filter parameters
        """
        if filter_params is None:
            # Default parameters optimized for BEP detection
            if filter_type == 'savgol':
                filter_params = {'window_length': 9, 'polyorder': 3}
            elif filter_type == 'gaussian':
                filter_params = {'sigma': 2.0}
            elif filter_type == 'median':
                filter_params = {'kernel_size': 7}
            elif filter_type == 'moving_average':
                filter_params = {'window': 7}
        
        super().__init__(filter_type, filter_params)
        self.rated_flow = rated_flow
        self.name = f"Linear Q/P ({filter_type})"
        
    def calculate(self, measurement) -> float:
        """Calculate linear Q/P proxy"""
        Q = measurement.flow
        P = measurement.power
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
        
        # Simple linear relationship: (Q/P) × PF
        proxy = (Q / P) * PF
        
        return proxy
    
    def get_name(self) -> str:
        return self.name


class secondOrderProxy(ProxyFunction):
    """Original (Q²/P) × PF proxy with filtering"""
    
    def __init__(self, filter_type: str = 'savgol', filter_params: dict = None):
        if filter_params is None:
            if filter_type == 'savgol':
                filter_params = {'window_length': 9, 'polyorder': 3}
        
        super().__init__(filter_type, filter_params)
        self.name = f"Original Q²/P ({filter_type})"
        
    def calculate(self, measurement) -> float:
        Q = measurement.flow
        P = measurement.power  
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
            
        return (Q**2 / P) * PF
    
    def get_name(self) -> str:
        return self.name


class VolumetricEfficiencyProxy(ProxyFunction):
    """Volumetric efficiency proxy with PF amplification and filtering"""
    
    def __init__(self, rated_flow: float = 100.0,
                 filter_type: str = 'savgol',
                 filter_params: dict = None):
        if filter_params is None:
            if filter_type == 'savgol':
                filter_params = {'window_length': 9, 'polyorder': 3}
        
        super().__init__(filter_type, filter_params)
        self.rated_flow = rated_flow
        self.name = f"Volumetric Efficiency ({filter_type})"
        
    def calculate(self, measurement) -> float:
        """Calculate volumetric efficiency proxy"""
        Q = measurement.flow
        P = measurement.power
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
        
        # Base volumetric efficiency (Q/√P reduces head bias)
        base_efficiency = Q / np.sqrt(P)
        
        # Power factor amplification
        pf_normalized = (PF - 0.6) / 0.35  # Normalize to 0-1 range
        pf_bonus = 1.0 + 0.5 * pf_normalized  # 1.0 to 1.5 multiplier
        
        # Final proxy
        proxy = base_efficiency * pf_bonus
        
        return proxy
    
    def get_name(self) -> str:
        return self.name


class NormalizedProxy(ProxyFunction):
    """Normalized (Q/√P) × PF proxy with filtering"""
    
    def __init__(self, rated_flow: float = 100.0,
                 filter_type: str = 'savgol',
                 filter_params: dict = None):
        if filter_params is None:
            if filter_type == 'savgol':
                filter_params = {'window_length': 9, 'polyorder': 3}
        
        super().__init__(filter_type, filter_params)
        self.rated_flow = rated_flow
        self.name = f"Normalized Q/√P ({filter_type})"
        
    def calculate(self, measurement) -> float:
        Q = measurement.flow
        P = measurement.power
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
            
        return (Q / np.sqrt(P)) * PF
    
    def get_name(self) -> str:
        return self.name


# ==============================================================================
# FILTER COMPARISON UTILITY
# ==============================================================================

def compare_filters(measurements: List, proxy_class=firstOrderProxy, 
                   filters: List[str] = ['none', 'savgol', 'gaussian', 'median', 'moving_average']):
    """
    Compare different filter types for BEP detection
    
    Args:
        measurements: List of pump measurements
        proxy_class: Proxy class to test
        filters: List of filter types to compare
        
    Returns:
        Dictionary with results for each filter
    """
    import matplotlib.pyplot as plt
    
    results = {}
    
    fig, axes = plt.subplots(len(filters), 1, figsize=(10, 3*len(filters)))
    if len(filters) == 1:
        axes = [axes]
    
    flows = np.array([m.flow for m in measurements])
    true_effs = np.array([m.true_efficiency for m in measurements])
    
    for i, filter_type in enumerate(filters):
        # Create proxy with specific filter
        proxy = proxy_class(filter_type=filter_type)
        
        # Calculate raw and filtered values
        raw_values, filtered_values = proxy.calculate_batch(measurements)
        
        # Find BEP
        bep_idx = np.argmax(filtered_values)
        bep_flow = flows[bep_idx]
        
        # Find true BEP
        true_bep_idx = np.argmax(true_effs)
        true_bep_flow = flows[true_bep_idx]
        
        # Calculate error
        error = abs(bep_flow - true_bep_flow)
        error_pct = (error / true_bep_flow) * 100
        
        results[filter_type] = {
            'bep_flow': bep_flow,
            'error': error,
            'error_pct': error_pct,
            'raw_values': raw_values,
            'filtered_values': filtered_values
        }
        
        # Plot
        ax = axes[i]
        ax.plot(flows, raw_values, 'o-', alpha=0.3, label='Raw', markersize=3)
        ax.plot(flows, filtered_values, 'b-', linewidth=2, label='Filtered')
        ax.axvline(bep_flow, color='b', linestyle='--', label=f'Proxy BEP: {bep_flow:.3f}')
        ax.axvline(true_bep_flow, color='r', linestyle='--', label=f'True BEP: {true_bep_flow:.3f}')
        
        ax.set_xlabel('Flow (m³/h)')
        ax.set_ylabel('Proxy Value')
        ax.set_title(f'{filter_type.upper()} Filter - Error: {error_pct:.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return results, fig


# ==============================================================================
# RECOMMENDED CONFIGURATIONS
# ==============================================================================

RECOMMENDED_FILTERS = {
    'low_noise': {
        'filter_type': 'none',
        'filter_params': {}
    },
    'medium_noise': {
        'filter_type': 'savgol',
        'filter_params': {'window_length': 7, 'polyorder': 2}
    },
    'high_noise': {
        'filter_type': 'savgol',
        'filter_params': {'window_length': 11, 'polyorder': 3}
    },
    'outliers': {
        'filter_type': 'median',
        'filter_params': {'kernel_size': 7}
    },
    'smooth_aggressive': {
        'filter_type': 'gaussian',
        'filter_params': {'sigma': 2.5}
    }
}


def create_proxy_with_noise_level(proxy_class, noise_level: str = 'medium_noise'):
    """
    Factory function to create proxy with recommended filter for noise level
    
    Args:
        proxy_class: Proxy class (firstOrderProxy, etc.)
        noise_level: 'low_noise', 'medium_noise', 'high_noise', 'outliers', 'smooth_aggressive'
    """
    config = RECOMMENDED_FILTERS.get(noise_level, RECOMMENDED_FILTERS['medium_noise'])
    return proxy_class(**config)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PROXY FUNCTIONS WITH FILTERING")
    print("="*80)
    print("\nAvailable filters:")
    print("  1. none           - No filtering (for clean data)")
    print("  2. savgol         - Savitzky-Golay filter (smooth + preserves peaks)")
    print("  3. gaussian       - Gaussian smoothing (aggressive smoothing)")
    print("  4. median         - Median filter (removes outliers)")
    print("  5. moving_average - Simple moving average")
    
    print("\nRecommended configurations:")
    for name, config in RECOMMENDED_FILTERS.items():
        print(f"  {name:20s}: {config['filter_type']:15s} {config['filter_params']}")
    
    print("\nExample usage:")
    print("  # Create proxy with medium noise filtering")
    print("  proxy = firstOrderProxy(filter_type='savgol',")
    print("                         filter_params={'window_length': 9, 'polyorder': 3})")
    print()
    print("  # Or use factory function")
    print("  proxy = create_proxy_with_noise_level(firstOrderProxy, 'high_noise')")
    print()
    print("  # Calculate for batch with filtering")
    print("  raw_values, filtered_values = proxy.calculate_batch(measurements)")
    print("="*80)