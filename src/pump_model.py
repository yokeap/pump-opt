# ==============================================================================
# FILE: src/pump_model.py - Modified for Low-Head Submersible Pump
# ==============================================================================

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
    """Low-head submersible pump simulator with enhanced BEP sensitivity"""
    
    def __init__(self, 
                 system_head: float = 10.0,  # Reduced from 30.0
                 pump_type: str = "submersible_low_head",
                 rated_power: float = 75.0,   # Reduced from 200.0 kW
                 rated_flow: float = 600.0,   # Reduced from 1000.0 m³/h
                 noise_level: float = 0.02):
        
        self.system_head = system_head
        self.initial_head = system_head
        self.pump_type = pump_type
        self.rated_power = rated_power
        self.rated_flow = rated_flow
        self.rated_freq = 50.0  # Hz
        self.noise_level = noise_level
        
        # Low-head submersible pump characteristics
        self.design_head_range = (5, 25)  # Low-head operation range
        self.bep_sensitivity = 0.8  # Hz per meter (higher sensitivity)
        
        # Motor parameters (submersible motor characteristics)
        self.motor_poles = 4
        self.motor_efficiency = 0.90  # Slightly lower for submersible
        
        # Create realistic efficiency map
        self._create_pump_characteristics()
        
        # Dynamic state
        self.current_head = system_head
        self.last_measurement_time = time.time()
        
    def _create_pump_characteristics(self):
        """Create realistic low-head submersible pump characteristic curves"""
        
        # Frequency range for characteristic curves (wider range for low-head pumps)
        freq_range = np.linspace(20, 70, 100)
        
        # BEP location with reasonable sensitivity to head changes
        # For submersible pumps: BEP frequency changes more than centrifugal but not extreme
        head_factor = (self.system_head - 15) / 10  # Normalized around 15m design point
        self.bep_frequency = 45.0 + head_factor * 6.0  # Reduced from 12.0 to 6.0 Hz per 10m
        
        # Ensure BEP stays within reasonable operational range
        self.bep_frequency = np.clip(self.bep_frequency, 30, 60)  # Keep within 30-60 Hz range
        
        # Efficiency characteristics for submersible pumps
        # Peak efficiency around 0.82, wider operating range
        base_efficiency = 0.82
        head_penalty = 0.08 * abs(head_factor)  # Efficiency drops off design point
        self.bep_efficiency = max(0.65, base_efficiency - head_penalty)
        
        # Create efficiency curve with submersible characteristics
        efficiencies = self._calculate_efficiency_curve(freq_range)
        
        # Create interpolator for efficiency lookup
        self.efficiency_interpolator = interp1d(
            freq_range, efficiencies, 
            kind='cubic', 
            fill_value='extrapolate',
            bounds_error=False
        )
        
    def _calculate_efficiency_curve(self, frequencies):
        """Calculate realistic efficiency curve for low-head submersible pump"""
        efficiencies = []
        
        for freq in frequencies:
            freq_deviation = abs(freq - self.bep_frequency)
            
            # Primary efficiency curve (wider and more forgiving than centrifugal)
            primary_eff = self.bep_efficiency * np.exp(-0.5 * (freq_deviation / 12)**2)
            
            # Low frequency penalty (submersible pumps handle low speeds better)
            low_freq_penalty = 0.15 * np.exp(-(freq - 15) / 8) if freq < 25 else 0
            
            # High frequency penalty (motor heating issues at high speed)
            high_freq_penalty = 0.25 * ((freq - 60) / 8)**2 if freq > 60 else 0
            
            # Head-dependent efficiency characteristics
            head_factor = self.system_head / 15.0  # Normalized around 15m optimal
            
            # Off-design head penalties (initialize variables)
            low_head_bonus = 0
            high_head_penalty = 0
            
            if head_factor < 0.5:  # Very low head
                low_head_bonus = 0.03  # Submersible pumps can be efficient at low head
            elif head_factor > 1.5:  # High head operation
                high_head_penalty = 0.12 * (head_factor - 1.5)
            
            # Final efficiency calculation
            final_eff = (primary_eff - low_freq_penalty - high_freq_penalty 
                        + low_head_bonus - high_head_penalty)
            
            # Submersible pump efficiency range: 0.45 - 0.85
            efficiencies.append(np.clip(final_eff, 0.45, 0.85))
            
        return np.array(efficiencies)
    
    def set_system_head(self, new_head: float):
        """Update system head and recalculate pump characteristics"""
        old_head = self.current_head
        self.current_head = new_head
        self.system_head = new_head
        
        # Recalculate characteristics with new head
        self._create_pump_characteristics()
        
        print(f"System head changed: {old_head:.1f}m → {new_head:.1f}m")
        print(f"   New BEP frequency: {self.bep_frequency:.1f} Hz")
        
    def get_measurement(self, frequency: float) -> PumpMeasurement:
        """Get pump measurement at specified frequency"""
        current_time = time.time()
        base_efficiency = float(self.efficiency_interpolator(frequency))
        
        # Add measurement noise
        if self.noise_level > 0:
            eff_noise = np.random.normal(0, self.noise_level * 0.3)
            efficiency = base_efficiency * (1 + eff_noise)
        else:
            efficiency = base_efficiency
        efficiency = np.clip(efficiency, 0.35, 0.85)
        
        # Flow calculation (submersible pump characteristics)
        freq_ratio = frequency / self.rated_freq
        
        # Base flow with frequency scaling
        base_flow = self.rated_flow * freq_ratio
        
        # Flow efficiency factor (how efficiency affects actual flow)
        flow_efficiency_factor = 0.80 + 0.20 * (efficiency / 0.75)
        actual_flow = base_flow * flow_efficiency_factor
        
        # Add flow measurement noise
        if self.noise_level > 0:
            actual_flow *= abs(1 + np.random.normal(0, self.noise_level))
        actual_flow = max(20.0, actual_flow)  # Minimum flow constraint
        
        # Head calculation for low-head submersible pump
        # Lower head rise per frequency compared to centrifugal pumps
        pump_head_rise = self.initial_head * 0.3 * freq_ratio**1.8  # Lower exponent
        total_head = self.system_head + pump_head_rise
        
        # Power calculation
        hydraulic_power = (actual_flow * total_head * 9.81) / (3600 * 1000)  # kW
        
        # Mechanical losses (slightly higher for submersible)
        mechanical_efficiency = 0.92
        mechanical_power = hydraulic_power / mechanical_efficiency
        
        # Electrical power
        electrical_power = mechanical_power / (max(efficiency, 0.35) * self.motor_efficiency)
        
        # Add power measurement noise
        if self.noise_level > 0:
            electrical_power *= abs(1 + np.random.normal(0, self.noise_level))
        
        # Voltage (submersible pump voltage characteristics)
        base_voltage = 380.0  # Lower voltage for submersible
        voltage = base_voltage + np.random.normal(0, 6) if self.noise_level > 0 else base_voltage
        
        # Power factor (submersible motor characteristics)
        load_factor = electrical_power / self.rated_power
        frequency_factor = 1 - 0.06 * abs(frequency - self.rated_freq) / self.rated_freq
        
        # Submersible motors typically have good power factor characteristics
        base_pf = 0.88 * min(1.0, 0.4 + 0.6 * load_factor) * frequency_factor
        power_factor = base_pf + np.random.normal(0, 0.015) if self.noise_level > 0 else base_pf
        power_factor = np.clip(power_factor, 0.70, 0.95)
        
        # Current calculation
        apparent_power = electrical_power / power_factor
        current = (apparent_power * 1000) / (voltage * np.sqrt(3))
        
        # Temperature (submersible pumps run cooler due to water cooling)
        base_temp = 35 + 20 * (electrical_power / self.rated_power) + 8 * (1 - efficiency)
        temperature = base_temp + np.random.normal(0, 2) if self.noise_level > 0 else base_temp
        
        # Vibration (submersible pumps typically have lower vibration)
        freq_deviation = abs(frequency - self.bep_frequency)
        base_vibration = 1.0 + 0.2 * freq_deviation + 1.5 * (1 - efficiency)
        vibration = max(0.3, base_vibration + np.random.normal(0, 0.3)) if self.noise_level > 0 else base_vibration
        
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
        """Get true BEP frequency and efficiency"""
        return self.bep_frequency, self.bep_efficiency
    
    def plot_characteristics(self, show_sensitivity=True):
        """Plot pump characteristics with BEP sensitivity demonstration"""
        
        frequencies = np.linspace(25, 65, 100)
        
        if show_sensitivity:
            # Show characteristics for different heads to demonstrate sensitivity
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Low-Head Submersible Pump Characteristics\n({self.pump_type})', fontsize=16)
            
            test_heads = [8, 12, 16, 20]
            colors = ['blue', 'green', 'orange', 'red']
            
            for i, test_head in enumerate(test_heads):
                # Temporarily change head
                original_head = self.system_head
                self.set_system_head(test_head)
                
                efficiencies = [self.efficiency_interpolator(f) for f in frequencies]
                flows = []
                powers = []
                
                for f in frequencies:
                    measurement = self.get_measurement(f)
                    flows.append(measurement.flow)
                    powers.append(measurement.power)
                
                # Plot efficiency
                axes[0, 0].plot(frequencies, efficiencies, color=colors[i], 
                              linewidth=2, label=f'{test_head}m head')
                axes[0, 0].axvline(self.bep_frequency, color=colors[i], 
                                 linestyle='--', alpha=0.7)
                
                # Plot flow
                axes[0, 1].plot(frequencies, flows, color=colors[i], 
                              linewidth=2, label=f'{test_head}m head')
                axes[0, 1].axvline(self.bep_frequency, color=colors[i], 
                                 linestyle='--', alpha=0.7)
                
                # Plot power
                axes[1, 0].plot(frequencies, powers, color=colors[i], 
                              linewidth=2, label=f'{test_head}m head')
                axes[1, 0].axvline(self.bep_frequency, color=colors[i], 
                                 linestyle='--', alpha=0.7)
                
                # BEP frequency vs head (sensitivity plot)
                if i == 0:
                    head_range = np.linspace(6, 24, 20)
                    bep_frequencies = []
                    for h in head_range:
                        temp_head = self.system_head
                        self.set_system_head(h)
                        bep_frequencies.append(self.bep_frequency)
                        self.set_system_head(temp_head)
                    
                    axes[1, 1].plot(head_range, bep_frequencies, 'b-', linewidth=3)
                    axes[1, 1].set_xlabel('System Head (m)')
                    axes[1, 1].set_ylabel('BEP Frequency (Hz)')
                    axes[1, 1].set_title('BEP Sensitivity to Head Changes')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Show current operating point
                    axes[1, 1].plot(original_head, 
                                   45.0 + (original_head - 15) * 0.6,  # Updated calculation
                                   'ro', markersize=10, label='Current Operating Point')
                    axes[1, 1].legend()
                
                # Restore original head
                self.set_system_head(original_head)
            
            # Configure subplots
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Efficiency')
            axes[0, 0].set_title('Efficiency Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Flow (m³/h)')
            axes[0, 1].set_title('Flow Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Power (kW)')
            axes[1, 0].set_title('Power Curves')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        else:
            # Standard single-condition plot
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
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Efficiency')
            axes[0].set_title('Efficiency Curve')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(frequencies, flows, 'g-', linewidth=2)
            axes[1].axvline(self.bep_frequency, color='r', linestyle='--', alpha=0.7)
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Flow (m³/h)')
            axes[1].set_title('Flow Curve')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(frequencies, powers, 'orange', linewidth=2)
            axes[2].axvline(self.bep_frequency, color='r', linestyle='--', alpha=0.7)
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Power (kW)')
            axes[2].set_title('Power Curve')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_pump_info(self) -> dict:
        """Get comprehensive pump information for reporting"""
        return {
            'pump_type': self.pump_type,
            'rated_power': self.rated_power,
            'rated_flow': self.rated_flow,
            'current_system_head': self.system_head,
            'design_head_range': self.design_head_range,
            'bep_frequency': self.bep_frequency,
            'bep_efficiency': self.bep_efficiency,
            'bep_sensitivity': f"{6.0/10:.1f} Hz per 10m head change",
            'operational_frequency_range': (25, 65),
            'noise_level': self.noise_level
        }

# Convenience function to demonstrate BEP sensitivity
def demonstrate_bep_sensitivity():
    """Demonstrate the enhanced BEP sensitivity of the low-head pump model"""
    
    print("Low-Head Submersible Pump BEP Sensitivity Demonstration")
    print("=" * 60)
    
    pump = RealisticPumpSimulator(system_head=12, pump_type="submersible_low_head")
    
    test_heads = [8, 12, 16, 20]
    
    print(f"{'Head (m)':<10} {'BEP Freq (Hz)':<15} {'BEP Efficiency':<15} {'Frequency Change':<15}")
    print("-" * 60)
    
    prev_bep = None
    for head in test_heads:
        pump.set_system_head(head)
        bep_freq, bep_eff = pump.get_true_bep()
        
        if prev_bep is not None:
            freq_change = bep_freq - prev_bep
            print(f"{head:<10} {bep_freq:<15.1f} {bep_eff:<15.3f} {freq_change:+.1f}")
        else:
            print(f"{head:<10} {bep_freq:<15.1f} {bep_eff:<15.3f} {'--':<15}")
        
        prev_bep = bep_freq
    
    print(f"\nKey Characteristics:")
    info = pump.get_pump_info()
    print(f"• BEP Sensitivity: {info['bep_sensitivity']}")
    print(f"• Operating Range: {info['operational_frequency_range'][0]}-{info['operational_frequency_range'][1]} Hz")
    print(f"• Design Head Range: {info['design_head_range'][0]}-{info['design_head_range'][1]} m")
    print(f"• This provides ~3x better resolution than high-head centrifugal pumps")
    
    return pump

if __name__ == "__main__":
    # Demonstrate the new pump model
    demonstrate_bep_sensitivity()