import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import matplotlib.pyplot as plt

@dataclass
class PumpMeasurement:
    """Structured pump measurement data"""
    timestamp: float
    frequency: float
    flow: float           # m³/h
    power: float          # kW
    voltage: float        # V
    current: float        # A
    power_factor: float   # dimensionless
    temperature: Optional[float] = None     # °C
    vibration: Optional[float] = None       # mm/s
    pressure_in: Optional[float] = None     # bar
    pressure_out: Optional[float] = None    # bar
    
    # Hidden information (for validation)
    true_efficiency: Optional[float] = None
    true_head: Optional[float] = None
    system_head: Optional[float] = None

class RealisticPumpSimulator:
    """Publication-quality pump simulator with realistic characteristics"""
    
    def __init__(self, 
                 system_head: float = 30.0,
                 pump_type: str = "centrifugal",
                #  rated_power: float = 7.5,  # kW
                #  rated_flow: float = 100.0,  # m³/h
                 rated_power: float = 200.0,  # kW
                 rated_flow: float = 1000.0,  # m³/h
                 noise_level: float = 0.02):
        
        self.system_head = system_head
        self.initial_head = system_head
        self.pump_type = pump_type
        self.rated_power = rated_power
        self.rated_flow = rated_flow
        self.rated_freq = 50.0  # Hz
        self.noise_level = noise_level
        
        # Motor parameters
        self.motor_poles = 4
        self.motor_efficiency = 0.92
        
        # Create realistic efficiency map
        self._create_pump_characteristics()
        
        # Dynamic state
        self.current_head = system_head
        self.last_measurement_time = time.time()
        
    def _create_pump_characteristics(self):
        """Create realistic pump characteristic curves"""
        
        # Frequency range for characteristic curves
        freq_range = np.linspace(20, 65, 100)
        
        # BEP location depends on system head (realistic behavior)
        self.bep_frequency = 42.0 + (self.system_head - 30) * 0.4
        self.bep_efficiency = 0.78 - 0.15 * abs(self.system_head - 30) / 30
        
        # Create efficiency curve
        efficiencies = self._calculate_efficiency_curve(freq_range)
        
        # Create interpolator for efficiency lookup
        self.efficiency_interpolator = interp1d(
            freq_range, efficiencies, 
            kind='cubic', 
            fill_value='extrapolate',
            bounds_error=False
        )
        
    def _calculate_efficiency_curve(self, frequencies):
        """Calculate realistic efficiency curve"""
        efficiencies = []
        
        for freq in frequencies:
            freq_deviation = abs(freq - self.bep_frequency)
            primary_eff = self.bep_efficiency * np.exp(-0.5 * (freq_deviation / 8)**2)
            low_freq_penalty = 0.3 * np.exp(-(freq - 20) / 5) if freq < 25 else 0
            high_freq_penalty = 0.2 * ((freq - 55) / 10)**1.5 if freq > 55 else 0
            head_factor = self.system_head / 30.0
            high_head_penalty = 0.05 * (head_factor - 1.2) if head_factor > 1.2 else 0
            final_eff = primary_eff - low_freq_penalty - high_freq_penalty - high_head_penalty
            efficiencies.append(np.clip(final_eff, 0.1, 0.85))
            
        return np.array(efficiencies)
    
    def set_system_head(self, new_head: float):
        old_head = self.current_head
        self.current_head = new_head
        self.system_head = new_head
        self._create_pump_characteristics()
        print(f"📍 System head changed: {old_head:.1f}m → {new_head:.1f}m")
        print(f"   New BEP frequency: {self.bep_frequency:.1f} Hz")
        
    def get_measurement(self, frequency: float) -> PumpMeasurement:
        current_time = time.time()
        base_efficiency = float(self.efficiency_interpolator(frequency))
        
        # Add noise
        if self.noise_level > 0:
            eff_noise = np.random.normal(0, self.noise_level * 0.3)
            efficiency = base_efficiency * (1 + eff_noise)
        else:
            efficiency = base_efficiency
        efficiency = np.clip(efficiency, 0.2, 0.85)
        
        # Flow
        freq_ratio = frequency / self.rated_freq
        base_flow = self.rated_flow * freq_ratio
        flow_efficiency_factor = 0.85 + 0.15 * (efficiency / 0.75)
        actual_flow = base_flow * flow_efficiency_factor
        if self.noise_level > 0:
            actual_flow *= abs(1 + np.random.normal(0, self.noise_level))
        actual_flow = max(10.0, actual_flow)
        
        # Head
        pump_head_rise = self.initial_head * 0.5 * freq_ratio**2
        total_head = self.system_head + pump_head_rise
        
        # Power
        hydraulic_power = (actual_flow * total_head * 9.81) / (3600 * 1000)  # kW
        mechanical_power = hydraulic_power / 0.95
        electrical_power = mechanical_power / (max(efficiency, 0.2) * self.motor_efficiency)
        if self.noise_level > 0:
            electrical_power *= abs(1 + np.random.normal(0, self.noise_level))
        
        # Voltage
        base_voltage = 400.0
        voltage = base_voltage + np.random.normal(0, 8) if self.noise_level > 0 else base_voltage
        
        # Power factor
        load_factor = electrical_power / self.rated_power
        frequency_factor = 1 - 0.08 * abs(frequency - self.rated_freq) / self.rated_freq
        base_pf = 0.85 * min(1.0, 0.3 + 0.7 * load_factor) * frequency_factor
        power_factor = base_pf + np.random.normal(0, 0.02) if self.noise_level > 0 else base_pf
        power_factor = np.clip(power_factor, 0.65, 0.95)
        
        # Current
        apparent_power = electrical_power / power_factor
        current = (apparent_power * 1000) / (voltage * np.sqrt(3))
        
        # Temperature
        base_temp = 45 + 25 * (electrical_power / self.rated_power) + 10 * (1 - efficiency)
        temperature = base_temp + np.random.normal(0, 3) if self.noise_level > 0 else base_temp
        
        # Vibration
        freq_deviation = abs(frequency - self.bep_frequency)
        base_vibration = 1.5 + 0.3 * freq_deviation + 2.0 * (1 - efficiency)
        vibration = max(0.5, base_vibration + np.random.normal(0, 0.5)) if self.noise_level > 0 else base_vibration
        
        return PumpMeasurement(
            timestamp=current_time,
            frequency=frequency,
            flow=actual_flow,
            power=electrical_power,
            voltage=voltage,
            current=current,
            power_factor=power_factor,
            temperature=temperature,
            vibration=vibration,
            true_efficiency=base_efficiency,
            true_head=total_head,
            system_head=self.current_head
        )
    
    def get_true_bep(self) -> Tuple[float, float]:
        return self.bep_frequency, self.bep_efficiency
    
    def plot_characteristics(self):
        frequencies = np.linspace(25, 60, 100)
        efficiencies = [self.efficiency_interpolator(f) for f in frequencies]
        flows = []
        powers = []
        
        for f in frequencies:
            measurement = self.get_measurement(f)
            flows.append(measurement.flow)
            powers.append(measurement.power)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(frequencies, efficiencies, 'b-', linewidth=2)
        axes[0].axvline(self.bep_frequency, color='r', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Frequency (Hz)'); axes[0].set_ylabel('Efficiency'); axes[0].set_title('Efficiency Curve'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(frequencies, flows, 'g-', linewidth=2)
        axes[1].axvline(self.bep_frequency, color='r', linestyle='--', alpha=0.7)
        axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('Flow (m³/h)'); axes[1].set_title('Flow Curve'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(frequencies, powers, 'orange', linewidth=2)
        axes[2].axvline(self.bep_frequency, color='r', linestyle='--', alpha=0.7)
        axes[2].set_xlabel('Frequency (Hz)'); axes[2].set_ylabel('Power (kW)'); axes[2].set_title('Power Curve'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
