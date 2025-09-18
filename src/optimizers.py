from abc import ABC, abstractmethod
import optuna
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from .pump_model import PumpMeasurement
from .proxy_functions import ProxyFunction

class BEPOptimizer(ABC):
    """Abstract base class for BEP optimization algorithms"""
    
    @abstractmethod
    def suggest_frequency(self) -> float:
        """Suggest next frequency to test"""
        pass
    
    @abstractmethod  
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update optimizer with measurement result"""
        pass
    
    @abstractmethod
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate (frequency, proxy_value)"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get optimizer name"""
        pass

class TPEOptimizer(BEPOptimizer):
    """TPE-based BEP optimizer using Optuna"""
    
    def __init__(self,
                 freq_min: float = 25.0,
                 freq_max: float = 60.0,
                 proxy_function: ProxyFunction = None,
                 max_freq_change: float = 5.0):
        
        self.freq_min = freq_min
        self.freq_max = freq_max  
        self.max_freq_change = max_freq_change
        self.proxy_function = proxy_function or VolumetricEfficiencyProxy()
        
        # TPE configuration optimized for pump optimization
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=3,      # Quick startup
            n_ei_candidates=24,      # Good exploration
            multivariate=True,
            warn_independent_sampling=False
        )
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"BEP_TPE_{id(self)}"
        )
        
        # History tracking
        self.history = []
        self.last_frequency = (freq_min + freq_max) / 2
        self.iteration = 0
        
    def suggest_frequency(self) -> float:
        """Suggest next frequency using TPE"""
        self.iteration += 1
        
        # Initial exploration phase
        if self.iteration <= 3:
            exploration_points = [0.3, 0.5, 0.7]
            fraction = exploration_points[self.iteration - 1]
            frequency = self.freq_min + fraction * (self.freq_max - self.freq_min)
        else:
            # TPE suggestion
            trial = self.study.ask()
            frequency = trial.suggest_float("frequency", self.freq_min, self.freq_max)
            
            # Apply rate limiting for safety
            if abs(frequency - self.last_frequency) > self.max_freq_change:
                if frequency > self.last_frequency:
                    frequency = self.last_frequency + self.max_freq_change
                else:
                    frequency = self.last_frequency - self.max_freq_change
                    
            # Store trial for later update
            self._current_trial = trial
        
        self.last_frequency = frequency
        return frequency
        
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update TPE with measurement result"""
        
        # Calculate proxy value
        proxy_value = self.proxy_function.calculate(measurement)
        
        # Store in history
        self.history.append({
            'iteration': self.iteration,
            'frequency': frequency,
            'proxy_value': proxy_value,
            'measurement': measurement,
            'true_efficiency': measurement.true_efficiency
        })
        
        # Update TPE study (skip initial exploration)
        if self.iteration > 3 and hasattr(self, '_current_trial'):
            self.study.tell(self._current_trial, proxy_value)
            
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate"""
        if not self.history:
            return None
            
        best_entry = max(self.history, key=lambda x: x['proxy_value'])
        return best_entry['frequency'], best_entry['proxy_value']
    
    def get_name(self) -> str:
        return f"TPE ({self.proxy_function.get_name()})"

class ExtremumSeekingControl(BEPOptimizer):
    """Classical Extremum Seeking Control for comparison"""
    
    def __init__(self, 
                 freq_min: float = 25.0,
                 freq_max: float = 60.0,
                 step_size: float = 2.0,
                 proxy_function: ProxyFunction = None):
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.initial_step = step_size
        self.step_size = step_size
        self.proxy_function = proxy_function or VolumetricEfficiencyProxy()
        
        # ESC state
        self.current_frequency = (freq_min + freq_max) / 2
        self.direction = 1  # 1 for increasing, -1 for decreasing
        self.history = []
        self.iteration = 0
        
        # Gradient estimation
        self.prev_proxy = None
        self.prev_frequency = None
        
    def suggest_frequency(self) -> float:
        """Suggest next frequency using ESC logic"""
        self.iteration += 1
        
        if self.iteration == 1:
            return self.current_frequency
            
        # Estimate gradient if we have previous data
        if self.prev_proxy is not None and self.prev_frequency is not None:
            freq_diff = self.current_frequency - self.prev_frequency
            if abs(freq_diff) > 1e-6:
                gradient = (self.history[-1]['proxy_value'] - self.prev_proxy) / freq_diff
                
                # Update direction based on gradient
                if gradient > 0:
                    # Keep current direction
                    pass
                else:
                    # Reverse direction
                    self.direction *= -1
                    self.step_size *= 0.9  # Reduce step size when oscillating
        
        # Calculate next frequency
        next_freq = self.current_frequency + self.direction * self.step_size
        
        # Boundary handling
        if next_freq <= self.freq_min:
            next_freq = self.freq_min + 1
            self.direction = 1
            self.step_size = self.initial_step * 0.5
        elif next_freq >= self.freq_max:
            next_freq = self.freq_max - 1
            self.direction = -1  
            self.step_size = self.initial_step * 0.5
            
        self.current_frequency = next_freq
        return next_freq
        
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update ESC with measurement result"""
        
        proxy_value = self.proxy_function.calculate(measurement)
        
        # Store previous values for gradient estimation
        if self.history:
            self.prev_proxy = self.history[-1]['proxy_value']
            self.prev_frequency = self.history[-1]['frequency']
        
        self.history.append({
            'iteration': self.iteration,
            'frequency': frequency,
            'proxy_value': proxy_value,
            'measurement': measurement,
            'true_efficiency': measurement.true_efficiency
        })
        
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate"""
        if not self.history:
            return None
            
        best_entry = max(self.history, key=lambda x: x['proxy_value'])
        return best_entry['frequency'], best_entry['proxy_value']
        
    def get_name(self) -> str:
        return f"ESC ({self.proxy_function.get_name()})"

class RandomSearch(BEPOptimizer):
    """Random search baseline for comparison"""
    
    def __init__(self,
                 freq_min: float = 25.0, 
                 freq_max: float = 60.0,
                 proxy_function: ProxyFunction = None,
                 seed: int = 42):
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.proxy_function = proxy_function or VolumetricEfficiencyProxy()
        self.rng = np.random.RandomState(seed)
        
        self.history = []
        self.iteration = 0
        
    def suggest_frequency(self) -> float:
        """Random frequency suggestion"""
        self.iteration += 1
        return self.rng.uniform(self.freq_min, self.freq_max)
        
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update with measurement result"""
        proxy_value = self.proxy_function.calculate(measurement)
        
        self.history.append({
            'iteration': self.iteration,
            'frequency': frequency, 
            'proxy_value': proxy_value,
            'measurement': measurement,
            'true_efficiency': measurement.true_efficiency
        })
        
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate"""
        if not self.history:
            return None
            
        best_entry = max(self.history, key=lambda x: x['proxy_value'])
        return best_entry['frequency'], best_entry['proxy_value']
        
    def get_name(self) -> str:
        return f"Random ({self.proxy_function.get_name()})"

class GridSearch(BEPOptimizer):
    """Systematic grid search for comparison"""
    
    def __init__(self,
                 freq_min: float = 25.0,
                 freq_max: float = 60.0,
                 n_points: int = 25,
                 proxy_function: ProxyFunction = None):
        
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.proxy_function = proxy_function or VolumetricEfficiencyProxy()
        
        # Create grid points
        self.grid_points = np.linspace(freq_min, freq_max, n_points)
        self.current_index = 0
        
        self.history = []
        self.iteration = 0
        
    def suggest_frequency(self) -> float:
        """Suggest next grid point"""
        self.iteration += 1
        
        if self.current_index < len(self.grid_points):
            freq = self.grid_points[self.current_index]
            self.current_index += 1
            return freq
        else:
            # Grid exhausted, return best found so far
            if self.history:
                best_entry = max(self.history, key=lambda x: x['proxy_value'])
                return best_entry['frequency']
            else:
                return (self.freq_min + self.freq_max) / 2
                
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update with measurement result"""
        proxy_value = self.proxy_function.calculate(measurement)
        
        self.history.append({
            'iteration': self.iteration,
            'frequency': frequency,
            'proxy_value': proxy_value, 
            'measurement': measurement,
            'true_efficiency': measurement.true_efficiency
        })
        
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate"""
        if not self.history:
            return None
            
        best_entry = max(self.history, key=lambda x: x['proxy_value'])
        return best_entry['frequency'], best_entry['proxy_value']
        
    def get_name(self) -> str:
        return f"Grid ({self.proxy_function.get_name()})"