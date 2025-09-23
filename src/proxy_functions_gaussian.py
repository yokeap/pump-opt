# ==============================================================================
# FILE: src/proxy_functions_gaussian.py
# Enhanced with Data-Driven Gaussian Method
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
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

class DataDrivenGaussianProxy(ProxyFunction):
    """
    Data-driven Gaussian proxy function using measurement sweep training
    
    η_proxy(Q) = η_max * exp(-2σ²(Q - Q_BEP)²)
    
    This proxy learns the BEP location from a training sweep and then
    provides accurate efficiency estimation based on flow rate.
    """
    
    def __init__(self, 
                 rated_flow: float = 1000.0,
                 training_required: bool = True,
                 fallback_proxy: Optional[ProxyFunction] = None):
        """
        Initialize Gaussian proxy
        
        Args:
            rated_flow: Pump rated flow rate (m³/h)
            training_required: Whether training sweep is required before use
            fallback_proxy: Fallback proxy to use before training
        """
        self.rated_flow = rated_flow
        self.training_required = training_required
        self.fallback_proxy = fallback_proxy
        
        # Gaussian model parameters (learned from training)
        self.eta_max = None
        self.Q_bep = None
        self.sigma = None
        self.is_trained = False
        
        # Training data storage
        self.training_flows = []
        self.training_proxies = []
        self.training_measurements = []
        
        # Model quality metrics
        self.fit_quality = None
        self.training_r_squared = None

    def calculate_from_flow(self, Q_arr):
        """
        Calculate efficiency from flow using Gaussian proxy
        Q_arr: float or np.array
        """
        if self.Q_bep is None or self.eta_max is None or self.sigma is None:
            raise ValueError("Proxy not trained yet: Q_bep, eta_max, sigma must be set.")
        Q_arr = np.array(Q_arr, dtype=float)
        eta_pred = self.eta_max * np.exp(-2*self.sigma**2*(Q_arr - self.Q_bep)**2)
        return eta_pred
        
    def train_from_sweep(self, 
                        measurements: List[PumpMeasurement], 
                        frequency_range: Tuple[float, float] = (25, 60),
                        n_points: int = 20,
                        proxy_method: str = 'volumetric') -> Dict[str, Any]:
        """
        Train Gaussian model from measurement sweep
        
        Args:
            measurements: List of measurements across frequency range
            frequency_range: Frequency sweep range (Hz)
            n_points: Number of measurement points
            proxy_method: Base proxy method ('volumetric', 'original', or 'power')
            
        Returns:
            Training results and model quality metrics
        """
        
        print(f"Training Gaussian proxy from {len(measurements)} measurements...")
        
        # Extract flow rates and calculate base proxy values
        flows = []
        proxy_values = []
        
        for measurement in measurements:
            Q = measurement.flow
            
            # Calculate base proxy value using specified method
            if proxy_method == 'volumetric':
                # Volumetric efficiency: Q/√P with PF amplification
                P = measurement.power
                PF = measurement.power_factor
                if P > 0:
                    base_proxy = (Q / np.sqrt(P)) * (1.0 + 0.5 * (PF - 0.6) / 0.35)
                else:
                    base_proxy = -100.0
                    
            elif proxy_method == 'original':
                # Original: (Q²/P) × PF
                P = measurement.power
                PF = measurement.power_factor
                if P > 0:
                    base_proxy = (Q**2 / P) * PF
                else:
                    base_proxy = -100.0
                    
            elif proxy_method == 'power':
                # Power-based: 1/P (simple inverse power)
                P = measurement.power
                if P > 0:
                    base_proxy = 1.0 / P
                else:
                    base_proxy = -100.0
                    
            else:
                raise ValueError(f"Unknown proxy method: {proxy_method}")
            
            flows.append(Q)
            proxy_values.append(base_proxy)
        
        self.training_flows = flows
        self.training_proxies = proxy_values
        self.training_measurements = measurements
        
        # Fit Gaussian model
        try:
            # Initial parameter guess
            max_proxy_idx = np.argmax(proxy_values)
            eta_max_guess = np.max(proxy_values)
            Q_bep_guess = flows[max_proxy_idx]
            sigma_guess = (np.max(flows) - np.min(flows)) / 6  # Reasonable spread
            
            initial_guess = [eta_max_guess, Q_bep_guess, sigma_guess]
            
            # Gaussian fitting function
            def gaussian_model(Q, eta_max, Q_bep, sigma):
                return eta_max * np.exp(-2 * sigma**2 * (Q - Q_bep)**2)
            
            # Fit the model
            popt, pcov = curve_fit(gaussian_model, flows, proxy_values, 
                                 p0=initial_guess, 
                                 maxfev=2000,
                                 bounds=([0, np.min(flows), 0.001], [np.inf, np.max(flows), np.ptp(flows)]))
            
            self.eta_max, self.Q_bep, self.sigma = popt
            
            # Calculate fit quality
            predicted = [gaussian_model(q, *popt) for q in flows]
            ss_res = np.sum((np.array(proxy_values) - np.array(predicted))**2)
            ss_tot = np.sum((np.array(proxy_values) - np.mean(proxy_values))**2)
            self.training_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.is_trained = True
            self.fit_quality = 'Good' if self.training_r_squared > 0.8 else 'Fair'
            
            # Training results
            training_results = {
                'success': True,
                'eta_max': self.eta_max,
                'Q_bep': self.Q_bep,
                'sigma': self.sigma,
                'r_squared': self.training_r_squared,
                'fit_quality': self.fit_quality,
                'n_training_points': len(measurements),
                'Q_range': (np.min(flows), np.max(flows)),
                'proxy_method': proxy_method,
                'predicted_bep_frequency': self._estimate_bep_frequency(measurements),
                'training_data': {
                    'flows': flows,
                    'proxy_values': proxy_values,
                    'predicted_values': predicted
                }
            }
            
            print(f"✅ Training successful:")
            print(f"   η_max = {self.eta_max:.4f}")
            print(f"   Q_BEP = {self.Q_bep:.1f} m³/h") 
            print(f"   σ = {self.sigma:.4f}")
            print(f"   R² = {self.training_r_squared:.3f}")
            print(f"   Estimated BEP frequency: {training_results['predicted_bep_frequency']:.1f} Hz")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
            # Fallback to peak detection
            max_idx = np.argmax(proxy_values)
            self.eta_max = proxy_values[max_idx]
            self.Q_bep = flows[max_idx]
            self.sigma = 0.01  # Conservative sigma
            self.is_trained = False
            self.fit_quality = 'Failed'
            
            return {
                'success': False,
                'error': str(e),
                'fallback_used': True,
                'Q_bep_estimate': self.Q_bep,
                'eta_max_estimate': self.eta_max
            }
    
    def _estimate_bep_frequency(self, measurements: List[PumpMeasurement]) -> float:
        """Estimate BEP frequency from training measurements"""
        
        # Find measurement closest to predicted BEP flow rate
        if not measurements or self.Q_bep is None:
            return 50.0  # Default frequency
        
        closest_measurement = min(measurements, 
                                key=lambda m: abs(m.flow - self.Q_bep))
        return closest_measurement.frequency
    
    def calculate(self, measurement: PumpMeasurement) -> float:
        """Calculate Gaussian proxy value"""
        
        if not self.is_trained:
            if self.fallback_proxy is not None:
                # Use fallback proxy if available
                return self.fallback_proxy.calculate(measurement)
            else:
                # Simple volumetric proxy as fallback
                Q = measurement.flow
                P = measurement.power
                PF = measurement.power_factor
                
                if P > 0:
                    return (Q / np.sqrt(P)) * (1.0 + 0.5 * (PF - 0.6)/0.35)
                else:
                    return -100.0
        
        # Use trained Gaussian model
        Q = measurement.flow
        
        if self.eta_max is None or self.Q_bep is None or self.sigma is None:
            return -100.0
        
        proxy_value = self.eta_max * np.exp(-((Q - self.Q_bep)**2) / (2 * self.sigma**2))
        
        return proxy_value
    
    def predict_bep_flow(self) -> Optional[float]:
        """Get predicted BEP flow rate"""
        return self.Q_bep if self.is_trained else None
    
    def get_model_parameters(self) -> Dict[str, float]:
        """Get fitted model parameters"""
        return {
            'eta_max': self.eta_max,
            'Q_bep': self.Q_bep, 
            'sigma': self.sigma,
            'r_squared': self.training_r_squared,
            'is_trained': self.is_trained
        }
    
    def evaluate_model_quality(self) -> Dict[str, Any]:
        """Evaluate trained model quality"""
        
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        # Model parameters check
        param_quality = {
            'eta_max_reasonable': 0.1 < self.eta_max < 100.0,
            'Q_bep_in_range': 0.5 * self.rated_flow < self.Q_bep < 1.5 * self.rated_flow,
            'sigma_reasonable': 0.001 < self.sigma < 0.1,
            'r_squared_good': self.training_r_squared > 0.7
        }
        
        quality_score = sum(param_quality.values()) / len(param_quality)
        
        # Overall assessment
        if quality_score > 0.75 and self.training_r_squared > 0.8:
            overall_quality = 'Excellent'
        elif quality_score > 0.5 and self.training_r_squared > 0.6:
            overall_quality = 'Good'
        elif quality_score > 0.25:
            overall_quality = 'Fair'
        else:
            overall_quality = 'Poor'
        
        return {
            'overall_quality': overall_quality,
            'quality_score': quality_score,
            'r_squared': self.training_r_squared,
            'parameter_checks': param_quality,
            'model_parameters': self.get_model_parameters(),
            'recommendation': self._get_quality_recommendation(overall_quality)
        }
    
    def _get_quality_recommendation(self, quality: str) -> str:
        """Get recommendation based on model quality"""
        
        recommendations = {
            'Excellent': 'Model ready for production use',
            'Good': 'Model suitable for optimization with monitoring',
            'Fair': 'Consider retraining with more data or different proxy method',
            'Poor': 'Retrain model or use fallback proxy method'
        }
        
        return recommendations.get(quality, 'Unknown quality level')
    
    def plot_training_results(self, save_path: Optional[str] = None):
        """Plot training results and model fit"""
        
        if not self.training_flows:
            print("No training data available for plotting")
            return None
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Gaussian Proxy Training Results', fontsize=16)
        
        flows = np.array(self.training_flows)
        proxies = np.array(self.training_proxies)
        
        # 1. Training data and fitted curve
        ax1 = axes[0, 0]
        ax1.scatter(flows, proxies, alpha=0.7, s=30, label='Training Data')
        
        if self.is_trained:
            flow_smooth = np.linspace(np.min(flows), np.max(flows), 100)
            proxy_smooth = [self.eta_max * np.exp(-((q - self.Q_bep)**2) / (2 * self.sigma**2))
                            for q in flow_smooth]
            ax1.plot(flow_smooth, proxy_smooth, 'r-', linewidth=2, 
                    label=f'Gaussian Fit (R²={self.training_r_squared:.3f})')
            ax1.axvline(self.Q_bep, color='green', linestyle='--', 
                    label=f'BEP: {self.Q_bep:.1f} m³/h')
        
        ax1.set_xlabel('Flow Rate (m³/h)')
        ax1.set_ylabel('Proxy Value')
        ax1.set_title('(a) Model Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = axes[0, 1]
        if self.is_trained:
            predicted = [self.eta_max * np.exp(-((q - self.Q_bep)**2) / (2 * self.sigma**2)) 
                        for q in flows]
            residuals = proxies - np.array(predicted)
            ax2.scatter(flows, residuals, alpha=0.7, s=30)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_xlabel('Flow Rate (m³/h)')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'(b) Residuals (σ={np.std(residuals):.4f})')
        else:
            ax2.text(0.5, 0.5, 'Model not trained', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title('(b) Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter visualization
        ax3 = axes[1, 0]
        if self.is_trained:
            params = ['η_max', 'Q_BEP', 'σ']
            values = [self.eta_max, self.Q_bep, self.sigma]
            
            bars = ax3.bar(params, values, alpha=0.7)
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('(c) Model Parameters')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'Model not trained', ha='center', va='center', 
                    transform=ax3.transAxes)
            ax3.set_title('(c) Model Parameters')
        ax3.grid(True, alpha=0.3)
        
        # 4. Quality metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        quality_info = self.evaluate_model_quality()
        
        if self.is_trained:
            info_text = f"""
    MODEL QUALITY ASSESSMENT
    {'='*25}

    Overall Quality: {quality_info['overall_quality']}
    Quality Score: {quality_info['quality_score']:.2f}
    R-squared: {quality_info['r_squared']:.3f}

    PARAMETERS
    {'-'*12}
    η_max: {self.eta_max:.4f}
    Q_BEP: {self.Q_bep:.1f} m³/h  
    σ: {self.sigma:.4f}

    RECOMMENDATION
    {'-'*14}
    {quality_info['recommendation']}
    """
        else:
            info_text = """
    MODEL STATUS
    {'='*12}

    Status: Not Trained
    Training: Failed

    Use fallback proxy method
    or retrain with better data
    """
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training results plot saved: {save_path}")
        
        return fig
    
    def get_name(self) -> str:
        if self.is_trained:
            return f"Gaussian (Q_BEP={self.Q_bep:.1f}, R²={self.training_r_squared:.3f})"
        else:
            return "Gaussian (Untrained)"

# Enhanced training utilities

class GaussianProxyTrainer:
    """Utility class for training Gaussian proxy models"""
    
    def __init__(self, pump_simulator):
        """
        Initialize trainer
        
        Args:
            pump_simulator: Pump simulator for generating training data
        """
        self.pump_simulator = pump_simulator
        
    def generate_training_sweep(self, 
                              frequency_range: Tuple[float, float] = (25, 60),
                              n_points: int = 25,
                              noise_level: float = 0.01) -> List[PumpMeasurement]:
        """Generate training sweep across frequency range"""
        
        print(f"Generating training sweep: {frequency_range[0]}-{frequency_range[1]} Hz, {n_points} points")
        
        frequencies = np.linspace(frequency_range[0], frequency_range[1], n_points)
        measurements = []
        
        # Temporarily set noise level
        original_noise = self.pump_simulator.noise_level
        self.pump_simulator.noise_level = noise_level
        
        try:
            for freq in frequencies:
                measurement = self.pump_simulator.get_measurement(freq)
                measurements.append(measurement)
        finally:
            # Restore original noise level
            self.pump_simulator.noise_level = original_noise
        
        print(f"Generated {len(measurements)} training measurements")
        return measurements
    
    def train_multiple_proxies(self, 
                             proxy_methods: List[str] = ['volumetric', 'original', 'power'],
                             **sweep_kwargs) -> Dict[str, DataDrivenGaussianProxy]:
        """Train multiple Gaussian proxies with different base methods"""
        
        # Generate training data once
        measurements = self.generate_training_sweep(**sweep_kwargs)
        
        trained_proxies = {}
        
        for method in proxy_methods:
            print(f"\n🔧 Training Gaussian proxy with {method} base method...")
            
            proxy = DataDrivenGaussianProxy(
                rated_flow=self.pump_simulator.rated_flow,
                training_required=True
            )
            
            # Train the proxy
            training_results = proxy.train_from_sweep(
                measurements, 
                proxy_method=method
            )
            
            quality_eval = proxy.evaluate_model_quality()
            trained_proxies[f"gaussian_{method}"] = {
                'proxy': proxy,
                'training_results': training_results,
                'quality': quality_eval
            }
        
        return trained_proxies
    
    def compare_proxy_performance(self, 
                                trained_proxies: Dict[str, Dict],
                                test_frequencies: List[float] = None) -> Dict[str, Any]:
        """Compare performance of different trained proxies"""
        
        if test_frequencies is None:
            test_frequencies = np.linspace(30, 55, 15)  # Test around BEP region
        
        print(f"Comparing proxy performance on {len(test_frequencies)} test points...")
        
        comparison_results = {
            'test_frequencies': test_frequencies,
            'proxy_performance': {},
            'true_bep': self.pump_simulator.get_true_bep()
        }
        
        valid_proxies = 0
        
        for proxy_name, proxy_info in trained_proxies.items():
            proxy = proxy_info['proxy']
            
            # Skip failed proxies
            if proxy is None or not proxy_info['training_results'].get('success', False):
                print(f"Skipping failed proxy: {proxy_name}")
                continue
            
            valid_proxies += 1
            proxy_values = []
            true_efficiencies = []
            
            for freq in test_frequencies:
                try:
                    measurement = self.pump_simulator.get_measurement(freq)
                    proxy_val = proxy.calculate(measurement)
                    
                    proxy_values.append(proxy_val)
                    true_efficiencies.append(measurement.true_efficiency)
                except Exception as e:
                    print(f"Warning: Error evaluating {proxy_name} at {freq} Hz: {e}")
                    proxy_values.append(0.0)
                    true_efficiencies.append(0.5)
            
            # Calculate correlation (handle edge cases)
            try:
                if len(proxy_values) > 1 and len(set(proxy_values)) > 1:
                    correlation = np.corrcoef(proxy_values, true_efficiencies)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
            except:
                correlation = 0.0
            
            # Find predicted BEP (max proxy value)
            if proxy_values and max(proxy_values) > min(proxy_values):
                max_idx = np.argmax(proxy_values)
                predicted_bep = test_frequencies[max_idx]
            else:
                predicted_bep = test_frequencies[len(test_frequencies)//2]  # Middle frequency as fallback
            
            true_bep_freq = comparison_results['true_bep'][0]
            bep_error = abs(predicted_bep - true_bep_freq)
            
            # Get training quality safely
            training_quality = proxy_info['quality'].get('overall_quality', 'Unknown')
            if training_quality is None:
                training_quality = 'Unknown'
            
            comparison_results['proxy_performance'][proxy_name] = {
                'proxy_values': proxy_values,
                'true_efficiencies': true_efficiencies,
                'correlation': correlation,
                'predicted_bep_frequency': predicted_bep,
                'bep_error': bep_error,
                'max_proxy_value': max(proxy_values) if proxy_values else 0.0,
                'training_quality': training_quality
            }
        
        if valid_proxies == 0:
            print("Warning: No valid proxies available for comparison")
            comparison_results['rankings'] = {'by_correlation': [], 'by_bep_accuracy': []}
            return comparison_results
        
        # Rank proxies by correlation and BEP accuracy
        rankings = {}
        
        # By correlation
        rankings['by_correlation'] = sorted(
            comparison_results['proxy_performance'].items(),
            key=lambda x: x[1]['correlation'],
            reverse=True
        )
        
        # By BEP accuracy  
        rankings['by_bep_accuracy'] = sorted(
            comparison_results['proxy_performance'].items(),
            key=lambda x: x[1]['bep_error']
        )
        
        comparison_results['rankings'] = rankings
        
        # Print summary
        print(f"\nProxy Performance Comparison:")
        print(f"{'Proxy':<20} {'Correlation':<12} {'BEP Error':<10} {'Quality':<12}")
        print("-" * 55)
        
        for proxy_name, perf in comparison_results['proxy_performance'].items():
            print(f"{proxy_name:<20} {perf['correlation']:<12.3f} {perf['bep_error']:<10.2f} {perf['training_quality']:<12}")
        
        return comparison_results

# Example usage function
def demo_gaussian_proxy_training(pump_simulator, output_dir: str = "output"):
    """Demonstrate Gaussian proxy training workflow"""
    
    print("🎯 GAUSSIAN PROXY TRAINING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize trainer
    trainer = GaussianProxyTrainer(pump_simulator)
    
    # Train multiple proxy variants
    trained_proxies = trainer.train_multiple_proxies(
        proxy_methods=['volumetric', 'original', 'power'],
        frequency_range=(25, 60),
        n_points=30,
        noise_level=0.015
    )
    
    # Compare performance
    comparison = trainer.compare_proxy_performance(trained_proxies)
    
    # Plot results for best proxy
    best_proxy_name = comparison['rankings']['by_correlation'][0][0]
    best_proxy = trained_proxies[best_proxy_name]['proxy']
    
    print(f"\n🏆 Best proxy: {best_proxy_name}")
    print(f"   Correlation: {comparison['proxy_performance'][best_proxy_name]['correlation']:.3f}")
    print(f"   BEP Error: {comparison['proxy_performance'][best_proxy_name]['bep_error']:.2f} Hz")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training results
    fig = best_proxy.plot_training_results(
        save_path=f"{output_dir}/gaussian_proxy_training.png"
    )
    
    return {
        'trained_proxies': trained_proxies,
        'comparison_results': comparison,
        'best_proxy': best_proxy,
        'best_proxy_name': best_proxy_name
    }