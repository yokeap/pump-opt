# ==============================================================================
# FILE: src/optimizers.py
# ==============================================================================

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
    
    def suggest_frequencies(self, batch_size: int = 1) -> List[float]:
        """Suggest multiple frequencies (default implementation)"""
        return [self.suggest_frequency() for _ in range(batch_size)]
    
    @abstractmethod  
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update optimizer with measurement result"""
        pass
    
    def update_batch(self, frequencies: List[float], measurements: List[PumpMeasurement]):
        """Update optimizer with batch of measurements (default implementation)"""
        for freq, measurement in zip(frequencies, measurements):
            self.update(freq, measurement)
    
    @abstractmethod
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate (frequency, proxy_value)"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get optimizer name"""
        pass

class TPEOptimizerMiniBatch(BEPOptimizer):
    """Enhanced TPE-based BEP optimizer with mini-batch support using Optuna"""
    
    def __init__(self,
                 freq_min: float = 25.0,
                 freq_max: float = 60.0,
                 proxy_function: ProxyFunction = None,
                 max_freq_change: float = 5.0,
                 batch_size: int = 1,
                 exploration_ratio: float = 0.2):
        
        self.freq_min = freq_min
        self.freq_max = freq_max  
        self.max_freq_change = max_freq_change
        self.proxy_function = proxy_function or VolumetricEfficiencyProxy()
        self.batch_size = batch_size
        self.exploration_ratio = exploration_ratio
        
        # Enhanced TPE configuration for mini-batch
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(3, batch_size),  # At least batch_size startup trials
            n_ei_candidates=32,                    # Increased for better batch diversity
            multivariate=True,
            warn_independent_sampling=False,
            constant_liar=True if batch_size > 1 else False  # Important for parallel optimization
        )
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"BEP_TPE_Batch_{id(self)}"
        )
        
        # History tracking
        self.history = []
        self.last_frequencies = [(freq_min + freq_max) / 2]
        self.iteration = 0
        self.pending_trials = []  # Track trials awaiting results
        
        # Mini-batch state
        self.current_batch = []
        self.batch_counter = 0
        
    def suggest_frequency(self) -> float:
        """Single frequency suggestion (for backward compatibility)"""
        frequencies = self.suggest_frequencies(batch_size=1)
        return frequencies[0]
    
    def suggest_frequencies(self, batch_size: int = None) -> List[float]:
        """Suggest batch of frequencies using enhanced TPE"""
        if batch_size is None:
            batch_size = self.batch_size
            
        self.iteration += 1
        self.batch_counter += 1
        
        suggested_frequencies = []
        
        # Initial exploration phase - ensure good coverage
        if self.batch_counter <= max(2, int(np.ceil(6 / batch_size))):
            # Strategic exploration points
            exploration_fractions = np.linspace(0.1, 0.9, batch_size)
            np.random.shuffle(exploration_fractions)  # Add randomness
            
            for fraction in exploration_fractions:
                frequency = self.freq_min + fraction * (self.freq_max - self.freq_min)
                suggested_frequencies.append(frequency)
                
        else:
            # TPE-based batch suggestion
            trials = []
            
            for i in range(batch_size):
                trial = self.study.ask()
                trials.append(trial)
                
                # Get TPE suggestion
                frequency = trial.suggest_float("frequency", self.freq_min, self.freq_max)
                
                # Apply rate limiting for safety
                if self.last_frequencies:
                    closest_last_freq = min(self.last_frequencies, 
                                          key=lambda x: abs(x - frequency))
                    
                    if abs(frequency - closest_last_freq) > self.max_freq_change:
                        if frequency > closest_last_freq:
                            frequency = closest_last_freq + self.max_freq_change
                        else:
                            frequency = closest_last_freq - self.max_freq_change
                
                suggested_frequencies.append(frequency)
            
            # Ensure diversity in batch (avoid too similar frequencies)
            suggested_frequencies = self._ensure_batch_diversity(suggested_frequencies)
            
            # Store trials for later update
            self.pending_trials.extend(trials)
        
        # Update last frequencies
        self.last_frequencies = suggested_frequencies.copy()
        self.current_batch = suggested_frequencies.copy()
        
        return suggested_frequencies
    
    def _ensure_batch_diversity(self, frequencies: List[float], min_distance: float = 2.0) -> List[float]:
        """Ensure minimum distance between suggested frequencies"""
        if len(frequencies) <= 1:
            return frequencies
        
        diverse_frequencies = [frequencies[0]]
        
        for freq in frequencies[1:]:
            # Check distance from all previously selected frequencies
            min_dist = min(abs(freq - selected) for selected in diverse_frequencies)
            
            if min_dist >= min_distance:
                diverse_frequencies.append(freq)
            else:
                # Generate a new diverse frequency
                attempts = 0
                while attempts < 10:
                    new_freq = np.random.uniform(self.freq_min, self.freq_max)
                    min_dist_new = min(abs(new_freq - selected) for selected in diverse_frequencies)
                    
                    if min_dist_new >= min_distance:
                        diverse_frequencies.append(new_freq)
                        break
                    attempts += 1
                
                # If we can't find a diverse point, use the original
                if attempts >= 10:
                    diverse_frequencies.append(freq)
        
        return diverse_frequencies
        
    def update(self, frequency: float, measurement: PumpMeasurement):
        """Update TPE with single measurement result"""
        self.update_batch([frequency], [measurement])
    
    def update_batch(self, frequencies: List[float], measurements: List[PumpMeasurement]):
        """Update TPE with batch of measurement results"""
        
        proxy_values = []
        
        # Calculate proxy values for all measurements
        for freq, measurement in zip(frequencies, measurements):
            proxy_value = self.proxy_function.calculate(measurement)
            proxy_values.append(proxy_value)
            
            # Store in history
            self.history.append({
                'iteration': self.iteration,
                'batch': self.batch_counter,
                'frequency': freq,
                'proxy_value': proxy_value,
                'measurement': measurement,
                'true_efficiency': measurement.true_efficiency
            })
        
        # Update TPE study with batch results
        if self.batch_counter > max(2, int(np.ceil(6 / self.batch_size))) and self.pending_trials:
            # Match trials with results
            n_trials_to_update = min(len(self.pending_trials), len(proxy_values))
            
            for i in range(n_trials_to_update):
                trial = self.pending_trials.pop(0)
                proxy_value = proxy_values[i]
                
                # Update study with result
                self.study.tell(trial, proxy_value)
                
        # Adaptive batch size adjustment based on performance
        if len(self.history) >= 20:
            self._adapt_batch_size()
    
    def _adapt_batch_size(self):
        """Dynamically adjust batch size based on optimization progress"""
        if len(self.history) < 20:
            return
        
        # Calculate improvement rate over last few batches
        recent_history = self.history[-20:]
        recent_batches = {}
        
        for entry in recent_history:
            batch_id = entry['batch']
            if batch_id not in recent_batches:
                recent_batches[batch_id] = []
            recent_batches[batch_id].append(entry['proxy_value'])
        
        # Calculate batch-wise improvements
        if len(recent_batches) >= 3:
            batch_best = []
            for batch_values in recent_batches.values():
                batch_best.append(max(batch_values))
            
            # If improvement is stagnating, reduce batch size for exploitation
            recent_improvement = max(batch_best[-3:]) - max(batch_best[:-3]) if len(batch_best) >= 6 else 0
            
            if recent_improvement < 0.01 and self.batch_size > 1:
                self.batch_size = max(1, self.batch_size - 1)
            elif recent_improvement > 0.05 and self.batch_size < 5:
                self.batch_size += 1
                
    def get_best_bep(self) -> Optional[Tuple[float, float]]:
        """Get current best BEP estimate"""
        if not self.history:
            return None
            
        best_entry = max(self.history, key=lambda x: x['proxy_value'])
        return best_entry['frequency'], best_entry['proxy_value']
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get detailed convergence information"""
        if not self.history:
            return {}
        
        # Best value progression
        best_values = []
        current_best = -float('inf')
        
        for entry in self.history:
            current_best = max(current_best, entry['proxy_value'])
            best_values.append(current_best)
        
        # Batch statistics
        batch_stats = {}
        for entry in self.history:
            batch_id = entry['batch']
            if batch_id not in batch_stats:
                batch_stats[batch_id] = {'values': [], 'frequencies': []}
            batch_stats[batch_id]['values'].append(entry['proxy_value'])
            batch_stats[batch_id]['frequencies'].append(entry['frequency'])
        
        return {
            'total_evaluations': len(self.history),
            'total_batches': len(batch_stats),
            'current_batch_size': self.batch_size,
            'best_value': max(entry['proxy_value'] for entry in self.history),
            'best_frequency': self.get_best_bep()[0] if self.get_best_bep() else None,
            'convergence_curve': best_values,
            'batch_statistics': batch_stats
        }
    
    def get_name(self) -> str:
        return f"TPE-MiniBatch-{self.batch_size} ({self.proxy_function.get_name()})"

class TPEOptimizer(BEPOptimizer):
    """Original TPE-based BEP optimizer using Optuna (for backward compatibility)"""
    
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

# Convenience function to create optimizers
def create_optimizer(optimizer_type: str, **kwargs) -> BEPOptimizer:
    """Factory function to create optimizers"""
    
    optimizers = {
        'tpe': TPEOptimizer,
        'tpe_batch': TPEOptimizerMiniBatch,
        'esc': ExtremumSeekingControl,
        'random': RandomSearch,
        'grid': GridSearch
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}")
    
    return optimizers[optimizer_type](**kwargs)