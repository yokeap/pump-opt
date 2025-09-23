# ==============================================================================
# FILE: src/proxy_functions.py
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from .pump_model import PumpMeasurement

class ProxyFunction(ABC):
    """Abstract base class for efficiency proxy functions"""
    
    @abstractmethod
    def calculate(self, measurement: PumpMeasurement) -> float:
        """Calculate efficiency proxy from measurement"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get proxy function name"""
        pass

class VolumetricEfficiencyProxy(ProxyFunction):
    """Winner: Volumetric efficiency proxy with PF amplification"""
    
    def __init__(self, rated_flow: float = 100.0):
        self.rated_flow = rated_flow
        self.name = "Volumetric Efficiency"
        
    def calculate(self, measurement: PumpMeasurement) -> float:
        """Calculate volumetric efficiency proxy"""
        Q = measurement.flow
        P = measurement.power
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
        
        # Base volumetric efficiency (Q/√P reduces head bias)
        base_efficiency = Q / np.sqrt(P)
        
        # Power factor amplification (addresses PF limited contribution)
        pf_normalized = (PF - 0.6) / 0.35  # Normalize to 0-1 range
        pf_bonus = 1.0 + 0.5 * pf_normalized  # 1.0 to 1.5 multiplier
        
        # Final proxy
        proxy = base_efficiency * pf_bonus
        
        return proxy
    
    def get_name(self) -> str:
        return self.name

class OriginalProxy(ProxyFunction):
    """Original (Q²/P) × PF proxy for comparison"""
    
    def __init__(self):
        self.name = "Original Q²/P×PF"
        
    def calculate(self, measurement: PumpMeasurement) -> float:
        Q = measurement.flow
        P = measurement.power  
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
            
        return (Q**2 / P) * PF
    
    def get_name(self) -> str:
        return self.name

class NormalizedProxy(ProxyFunction):
    """Normalized (Q/√P) × PF proxy"""
    
    def __init__(self, rated_flow: float = 100.0):
        self.rated_flow = rated_flow
        self.name = "Volumetric Efficiency"
        
    def calculate(self, measurement: PumpMeasurement) -> float:
        Q = measurement.flow
        P = measurement.power
        PF = measurement.power_factor
        
        if Q <= 0 or P <= 0:
            return -100.0
            
        return (Q / np.sqrt(P)) * PF
    
    def get_name(self) -> str:
        return self.name