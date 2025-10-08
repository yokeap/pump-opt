# ==============================================================================
# FILE: src/commercial_pump_model.py
# Commercial Pump Model - Based on Schneider SUB 15-0.5cv
# Calibrated to: 60Hz, H=35m, Q=3m³/h, η=54%
# ==============================================================================

import numpy as np
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
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    pressure_in: Optional[float] = None
    pressure_out: Optional[float] = None
    
    # Hidden information (for validation)
    true_efficiency: Optional[float] = None
    true_head: Optional[float] = None
    system_head: Optional[float] = None


class CommercialPumpSimulator:
    """
    Commercial pump simulator based on Schneider SUB 15-0.5cv
    Calibrated operating point at 60Hz: H=35m, Q=3m³/h, η=54%
    
    Uses manufacturer's polynomial equations with adjusted coefficients
    """
    
    def __init__(self,
                 system_head: float = 30.0,
                 rated_power: float = 0.5,  # kW (0.5cv)
                 rated_flow: float = 3.0,   # m³/h (CORRECTED)
                 rated_head: float = 35.0,  # m (CORRECTED)
                 rated_eff: float = 0.54,   # 54% (CORRECTED)
                 rated_freq: float = 60.0,  # Hz
                 noise_level: float = 0.02,
                 random_seed: Optional[int] = None):
        
        self.system_head = system_head
        self.current_head = system_head
        self.rated_power = rated_power
        self.rated_flow = rated_flow
        self.rated_head = rated_head
        self.rated_eff = rated_eff
        self.rated_freq = rated_freq
        self.noise_level = noise_level
        
        # RNG for reproducible noise
        self.rng = np.random.default_rng(random_seed)
        
        # Motor efficiency (typical for small submersible pumps)
        self.motor_efficiency = 0.85
        
        # Recalibrated polynomial coefficients to match operating point
        # Head equation: H(Q,f) = c₀(f/60)² + c₁(f/60)Q + c₂Q²
        # Adjusted to pass through (Q=3, H=35) at f=60Hz
        self.head_coeffs = {
            'c0': 48.0,     # Shutoff head coefficient (f/60)²
            'c1': -2.8,     # Linear flow term (f/60)Q
            'c2': -0.85     # Quadratic flow term Q²
        }
        
        # Efficiency equation: η(Q,f) - calibrated for peak around 3 m³/h
        # η = a(Q-Q_opt)² + η_max at reference frequency
        # self.eff_coeffs = {
        #     'q_opt': 3.0,      # Optimal flow at 60Hz
        #     'eta_max': 0.54,   # Maximum efficiency (54%)
        #     'a': -0.015,       # Curvature coefficient
        # }

        self.eff_coeffs = {
            'a1': -0.361,      # Optimal flow at 60Hz
            'a2': -2.678,   # Maximum efficiency (54%)
            'a3': 29.433,       # Curvature coefficient
            'a4': 2.25,
        }
        
        # Operating limits
        self.min_frequency = 30.0  # Hz
        self.max_frequency = 65.0  # Hz
        self.max_flow = 5.0        # m³/h (adjusted for smaller pump)
        
        # Verify calibration
        self._verify_calibration()
        
        # Calculate BEP for current conditions
        self._find_bep()
        
        self.last_measurement_time = time.time()
    
    def _verify_calibration(self):
        """Verify model matches specified operating point"""
        H_check = self._calculate_head(self.rated_flow, self.rated_freq)
        eff_check = self._calculate_pump_efficiency(self.rated_flow, self.rated_freq)
        
        print(f"Calibration Check at 60Hz, Q={self.rated_flow} m³/h:")
        print(f"  Target Head: {self.rated_head:.1f}m, Calculated: {H_check:.1f}m")
        print(f"  Target Eff: {self.rated_eff*100:.1f}%, Calculated: {eff_check*100:.1f}%")
        
        # Fine-tune if needed
        if abs(H_check - self.rated_head) > 1.0:
            print("  Warning: Head calibration may need adjustment")
    
    def _calculate_head(self, flow: float, frequency: float) -> float:
        """Calculate pump head using calibrated equation"""
        f_ratio = frequency / 60.0
        
        H = (self.head_coeffs['c0'] * f_ratio**2 + 
             self.head_coeffs['c1'] * f_ratio * flow + 
             self.head_coeffs['c2'] * flow**2)
        
        return max(0.0, H)
    
    # ==============================================================================
    # FIXED METHOD for commercial_pump_model.py
    # Add this method to your CommercialPumpSimulator class
    # ==============================================================================

    def calculate_flow_from_frequency_head(self, frequency: float, target_head: float, 
                                        tolerance: float = 0.5) -> float:
        """
        FIXED: Calculate flow that produces target head at given frequency
        
        Solves: H(Q,f) = c₀(f/60)² + c₁(f/60)Q + c₂Q² = target_head
        Rearranges to: c₂Q² + c₁(f/60)Q + [c₀(f/60)² - target_head] = 0
        
        Args:
            frequency: Operating frequency in Hz
            target_head: Desired head in meters
            tolerance: Acceptable error in meters
            
        Returns:
            Flow rate in m³/h that produces target head, or None if not achievable
        """
        f_ratio = frequency / 60.0
        
        # Quadratic equation coefficients: aQ² + bQ + c = 0
        a = self.head_coeffs['c2']  # -0.85
        b = self.head_coeffs['c1'] * f_ratio  # -2.8 * (f/60)
        c = self.head_coeffs['c0'] * f_ratio**2 - target_head  # 48*(f/60)² - H
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # No real solution
            return None
        
        # Calculate both roots
        sqrt_discriminant = np.sqrt(discriminant)
        Q1 = (-b + sqrt_discriminant) / (2*a)
        Q2 = (-b - sqrt_discriminant) / (2*a)
        
        # Since c2 is negative (-0.85), parabola opens downward
        # We want the physically meaningful root in valid range
        candidates = []
        
        for Q in [Q1, Q2]:
            if 0.1 <= Q <= self.max_flow:
                # Verify this solution gives target head
                H_check = self._calculate_head(Q, frequency)
                error = abs(H_check - target_head)
                if error < tolerance:
                    candidates.append((Q, error))
        
        if not candidates:
            return None
        
        # Return the solution with smallest error
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
        
    def _calculate_pump_efficiency(self, flow: float, frequency: float) -> float:
        """
        Calculate pump hydraulic efficiency
        Uses parabolic model centered at optimal flow point
        Efficiency scales with frequency
        """
        if flow <= 0.01:
            return 0.01
        
        # Scale optimal flow with frequency
        # f_ratio = frequency / 60.0
        # q_opt_scaled = self.eff_coeffs['q_opt'] * f_ratio
        
        # # Parabolic efficiency curve
        # delta_q = flow - q_opt_scaled
        # eta = self.eff_coeffs['eta_max'] + self.eff_coeffs['a'] * delta_q**2
        
        # # Slight efficiency reduction at off-design frequencies
        # freq_factor = 1.0 - 0.05 * abs(1 - f_ratio)
        # eta = eta * freq_factor
        
        f_ratio = 60.0 / frequency
        eta = self.eff_coeffs['a1']*flow**3*f_ratio**3 + self.eff_coeffs['a2']*flow**2*f_ratio**2 + self.eff_coeffs['a3']*flow*f_ratio + self.eff_coeffs['a4']
        eta = eta * 0.01
        print(eta)
        
        return float(np.clip(eta, 0.10, 0.65))
    
    def _find_bep(self) -> None:
        """Find Best Efficiency Point at rated frequency"""
        flows = np.linspace(0.1, self.max_flow, 200)
        max_eff = 0.0
        self.bep_flow = self.rated_flow
        self.bep_efficiency = self.rated_eff
        
        for Q in flows:
            eff = self._calculate_pump_efficiency(Q, self.rated_freq)
            if eff > max_eff:
                max_eff = eff
                self.bep_flow = Q
                self.bep_efficiency = eff
        
        self.bep_head = self._calculate_head(self.bep_flow, self.rated_freq)
        
        print(f"\nBest Efficiency Point at {self.rated_freq}Hz:")
        print(f"  Flow: {self.bep_flow:.2f} m³/h")
        print(f"  Head: {self.bep_head:.2f} m")
        print(f"  Efficiency: {self.bep_efficiency*100:.1f}%")
    
    def _find_operating_point(self, frequency: float) -> Tuple[float, float, float]:
        """
        Find operating point where pump curve intersects system curve
        System curve: H_system = H_static + k*Q²
        
        Returns: (flow, head, efficiency)
        """
        # System resistance coefficient
        # Calculate k from known operating point
        if self.rated_flow > 0:
            k_system = (self.rated_head - self.system_head) / (self.rated_flow**2)
        else:
            k_system = 0.2
        
        # Ensure positive k
        k_system = max(0.1, k_system)
        
        # Solve for intersection: H_pump(Q,f) = H_system(Q)
        flows = np.linspace(0.1, self.max_flow, 200)
        min_diff = float('inf')
        best_flow = self.rated_flow * (frequency / self.rated_freq)
        
        for Q in flows:
            H_pump = self._calculate_head(Q, frequency)
            H_system = self.system_head + k_system * Q**2
            diff = abs(H_pump - H_system)
            
            if diff < min_diff and H_pump >= H_system:
                min_diff = diff
                best_flow = Q
        
        # Calculate operating point parameters
        operating_flow = best_flow
        operating_head = self._calculate_head(operating_flow, frequency)
        operating_efficiency = self._calculate_pump_efficiency(operating_flow, frequency)
        
        return operating_flow, operating_head, operating_efficiency
    
    def _calculate_electrical_power(self, flow: float, frequency: float) -> Tuple[float, float, float]:
        """
        Calculate electrical power consumption
        Returns: (power_kW, current_A, power_factor)
        """
        # Hydraulic power
        head = self._calculate_head(flow, frequency)
        pump_eff = self._calculate_pump_efficiency(flow, frequency)
        
        # P_hydraulic = ρ * g * Q * H (in kW)
        rho = 1000  # kg/m³
        g = 9.81    # m/s²
        Q_m3s = flow / 3600  # Convert m³/h to m³/s
        
        P_hydraulic = (rho * g * Q_m3s * head) / 1000  # kW
        
        # Electrical power accounting for efficiencies
        if pump_eff > 0.01 and self.motor_efficiency > 0:
            P_electrical = P_hydraulic / (pump_eff * self.motor_efficiency)
        else:
            P_electrical = 0.05  # Minimum idling power
        
        P_electrical = max(0.02, min(P_electrical, self.rated_power * 1.5))
        
        # Estimate current (assuming 3-phase 220V for small pump)
        voltage = 220.0  # More typical for 0.5kW pump
        power_factor = 0.85 - 0.15 * (1 - pump_eff)  # PF degrades with efficiency
        power_factor = np.clip(power_factor, 0.6, 0.92)
        
        current = (P_electrical * 1000) / (np.sqrt(3) * voltage * power_factor)
        
        return P_electrical, current, power_factor
    
    def measure(self, frequency: float, add_noise: bool = True) -> PumpMeasurement:
        """
        Simulate a measurement at given frequency
        
        Args:
            frequency: Operating frequency in Hz
            add_noise: Whether to add measurement noise
        
        Returns:
            PumpMeasurement object with all measured and hidden parameters
        """
        # Clip frequency to operating range
        frequency = np.clip(frequency, self.min_frequency, self.max_frequency)
        
        # Find operating point
        flow, head, efficiency = self._find_operating_point(frequency)
        
        # Calculate electrical parameters
        power, current, pf = self._calculate_electrical_power(flow, frequency)
        
        # Add measurement noise if requested
        if add_noise:
            flow = flow * (1 + self.rng.normal(0, self.noise_level))
            power = power * (1 + self.rng.normal(0, self.noise_level))
            current = current * (1 + self.rng.normal(0, self.noise_level * 0.5))
            pf = np.clip(pf + self.rng.normal(0, 0.02), 0.5, 1.0)
        
        # Voltage is relatively stable
        voltage = 220.0 + self.rng.normal(0, 3.0) if add_noise else 220.0
        
        # Optional sensors
        temperature = 45 + self.rng.normal(0, 3) if add_noise else 45.0
        vibration = 0.8 + self.rng.normal(0, 0.15) if add_noise else 0.8
        
        return PumpMeasurement(
            timestamp=time.time(),
            frequency=frequency,
            flow=max(0, flow),
            power=max(0, power),
            voltage=voltage,
            current=max(0, current),
            power_factor=pf,
            temperature=temperature,
            vibration=vibration,
            pressure_in=None,
            pressure_out=None,
            true_efficiency=efficiency,
            true_head=head,
            system_head=self.system_head
        )
    
    def set_system_head(self, new_head: float):
        """Change system operating conditions"""
        self.system_head = new_head
        self.current_head = new_head
    
    def get_true_bep(self) -> Tuple[float, float, float]:
        """
        Get the true Best Efficiency Point
        Returns: (flow, head, efficiency)
        """
        return self.bep_flow, self.bep_head, self.bep_efficiency
    
    def get_operating_point(self, frequency: float) -> Tuple[float, float, float]:
        """
        Get operating point at given frequency without noise
        Returns: (flow, head, efficiency)
        """
        return self._find_operating_point(frequency)
    
    def find_frequency_for_head(self, target_head: float, tolerance: float = 0.5) -> Optional[float]:
        """
        Find the frequency needed to achieve a target head
        
        Args:
            target_head: Desired head in meters
            tolerance: Acceptable error in meters
            
        Returns:
            Frequency in Hz, or None if target is unachievable
        """
        # Search through frequency range
        for freq in np.linspace(self.min_frequency, self.max_frequency, 100):
            flow, head, eff = self._find_operating_point(freq)
            if abs(head - target_head) <= tolerance:
                return freq
        
        return None
    
    def measure_at_head(self, target_head: float, add_noise: bool = True) -> Optional[PumpMeasurement]:
        """
        Set system to achieve target head and return measurement
        
        Args:
            target_head: Desired operating head in meters
            add_noise: Whether to add measurement noise
            
        Returns:
            PumpMeasurement object, or None if target head is unachievable
        """
        # Find frequency needed for this head
        freq = self.find_frequency_for_head(target_head)
        
        if freq is None:
            print(f"Warning: Target head {target_head}m is not achievable")
            print(f"  Frequency range: {self.min_frequency}-{self.max_frequency} Hz")
            return None
        
        # Temporarily update system head to match target
        old_system_head = self.system_head
        self.set_system_head(target_head - 0.5)  # Slight adjustment for system curve
        
        # Get measurement at this frequency
        measurement = self.measure(freq, add_noise=add_noise)
        
        # Restore original system head
        self.system_head = old_system_head
        
        return measurement
    
    def get_head_range(self) -> Tuple[float, float]:
        """
        Get the achievable head range for current system
        Returns: (min_head, max_head)
        """
        _, head_min, _ = self._find_operating_point(self.min_frequency)
        _, head_max, _ = self._find_operating_point(self.max_frequency)
        return head_min, head_max
    
    def measure_at_head_safe(self, target_head: float, add_noise: bool = True) -> PumpMeasurement:
        """
        Safe version that always returns a measurement by clamping head to valid range
        
        Args:
            target_head: Desired operating head in meters
            add_noise: Whether to add measurement noise
            
        Returns:
            PumpMeasurement object (always valid)
        """
        # Get achievable range
        head_min, head_max = self.get_head_range()
        
        # Clamp to valid range
        if target_head < head_min:
            print(f"Warning: Target head {target_head}m too low, using minimum {head_min:.1f}m")
            target_head = head_min
        elif target_head > head_max:
            print(f"Warning: Target head {target_head}m too high, using maximum {head_max:.1f}m")
            target_head = head_max
        
        # Now measure (guaranteed to succeed)
        measurement = self.measure_at_head(target_head, add_noise)
        
        # This should never be None now, but handle it just in case
        if measurement is None:
            # Fallback to rated conditions
            return self.measure(self.rated_freq, add_noise)
        
        return measurement
    
    def calculate_hydraulic_power(self, flow: float, head: float) -> float:
        """
        Calculate hydraulic power output
        P_hydraulic = ρ * g * Q * H
        
        Args:
            flow: Flow rate in m³/h
            head: Head in m
            
        Returns:
            Hydraulic power in kW
        """
        rho = 1000  # kg/m³
        g = 9.81    # m/s²
        Q_m3s = flow / 3600  # Convert m³/h to m³/s
        
        P_hydraulic = (rho * g * Q_m3s * head) / 1000  # kW
        return P_hydraulic
    
    def plot_pump_curves(self, frequencies: Optional[list] = None):
        """Plot pump performance curves"""
        if frequencies is None:
            frequencies = [30, 40, 50, 60]
        
        flows = np.linspace(0, self.max_flow, 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Head curves
        for freq in frequencies:
            heads = [self._calculate_head(Q, freq) for Q in flows]
            ax1.plot(flows, heads, label=f'{freq} Hz', linewidth=2)
        
        # Mark rated point
        ax1.plot(self.rated_flow, self.rated_head, 'r*', 
                markersize=15, label=f'Rated Point ({self.rated_freq}Hz)')
        
        ax1.set_xlabel('Flow (m³/h)', fontsize=11)
        ax1.set_ylabel('Head (m)', fontsize=11)
        ax1.set_title('Pump Head Curves - Schneider SUB 15-0.5cv', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.max_flow)
        
        # Efficiency curves
        for freq in frequencies:
            effs = [self._calculate_pump_efficiency(Q, freq) * 100 for Q in flows]
            ax2.plot(flows, effs, label=f'{freq} Hz', linewidth=2)
        
        # Mark rated point
        ax2.plot(self.rated_flow, self.rated_eff * 100, 'r*', 
                markersize=15, label=f'Rated Point ({self.rated_freq}Hz)')
        
        ax2.set_xlabel('Flow (m³/h)', fontsize=11)
        ax2.set_ylabel('Efficiency (%)', fontsize=11)
        ax2.set_title('Pump Efficiency Curves', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.max_flow)
        ax2.set_ylim(0, 70)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simulate the pump coefficients
    class TestPump:
        def __init__(self):
            self.head_coeffs = {'c0': 48.0, 'c1': -2.8, 'c2': -0.85}
            self.max_flow = 8.0
        
        def _calculate_head(self, flow, frequency):
            f_ratio = frequency / 60.0
            H = (self.head_coeffs['c0'] * f_ratio**2 + 
                 self.head_coeffs['c1'] * f_ratio * flow + 
                 self.head_coeffs['c2'] * flow**2)
            return max(0.0, H)
    
    # Test pump
    pump = CommercialPumpSimulator()
    
    # Add the fixed method to the class
    # TestPump.calculate_flow_from_frequency_head = pump,calculate_flow_from_frequency_head
    
    # Test at different frequencies for constant head
    target_head = 35.0
    frequencies = np.linspace(30, 65, 50)
    
    flows = []
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head)
        if flow is not None:
            flows.append(flow)
            # Verify
            actual_head = pump._calculate_head(flow, freq)
            print(f"f={freq:.1f}Hz → Q={flow:.3f}m³/h → H={actual_head:.2f}m (target={target_head}m)")
        else:
            flows.append(np.nan)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, flows, 'b-', linewidth=2)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Flow (m³/h)', fontsize=12)
    plt.title(f'Flow vs Frequency at Constant Head = {target_head}m', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nFlow range: {np.nanmin(flows):.2f} - {np.nanmax(flows):.2f} m³/h")
    print(f"Expected: Flow should INCREASE with frequency at constant head")