# ==============================================================================
# Proxy Function Validation Experiment: Q²/P vs True BEP Analysis
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from src.pump_model import RealisticPumpSimulator
from src.proxy_functions import OriginalProxy

class ProxyValidationExperiment:
    """Validate Q²/P proxy function against true BEP across different heads"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        print(f"Proxy validation results will be saved to: {self.output_dir.absolute()}")
    
    def run_validation_experiment(self, 
                                 test_heads: List[float] = [20, 30, 40,],
                                 freq_range: Tuple[float, float] = (25, 60),
                                 n_points: int = 50) -> Dict[str, Any]:
        """
        Comprehensive proxy validation experiment
        
        Args:
            test_heads: System heads to test [m]
            freq_range: Frequency sweep range (Hz)
            n_points: Number of frequency points to test
            
        Returns:
            Complete validation results including R², errors, and 3D surface data
        """
        
        print("\n" + "="*60)
        print("PROXY FUNCTION VALIDATION EXPERIMENT")
        print("="*60)
        print(f"Testing Q²/P proxy across heads: {test_heads}m")
        print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"Analysis points: {n_points} per head")
        
        all_results = []
        correlation_analysis = {}
        bep_analysis = {}
        surface_data = {'frequencies': [], 'flows': [], 'normalized_eff': [], 'heads': []}
        
        # Generate frequency sweep
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        
        for head in test_heads:
            print(f"\nAnalyzing Head: {head}m")
            print("-" * 30)
            
            # Setup pump for this head
            pump = RealisticPumpSimulator(system_head=head, noise_level=0.01)  # Low noise for clean data
            proxy_function = OriginalProxy()
            
            # Get true BEP for reference
            true_bep_freq, true_bep_eff = pump.get_true_bep()
            
            # Sweep across frequencies
            frequencies_data = []
            flows_data = []
            powers_data = []
            proxy_values = []
            true_efficiencies = []
            normalized_efficiencies = []
            
            print(f"  True BEP: {true_bep_freq:.1f} Hz, Efficiency: {true_bep_eff:.3f}")
            print("  Sweeping frequencies...")
            
            for freq in frequencies:
                # Get measurement
                measurement = pump.get_measurement(freq)
                
                # Calculate proxy value
                proxy_val = proxy_function.calculate(measurement)
                
                # Normalize efficiency (0-1 scale within this head's range)
                norm_eff = measurement.true_efficiency / true_bep_eff if true_bep_eff > 0 else 0
                
                # Store data
                frequencies_data.append(freq)
                flows_data.append(measurement.flow)
                powers_data.append(measurement.power)
                proxy_values.append(proxy_val)
                true_efficiencies.append(measurement.true_efficiency)
                normalized_efficiencies.append(norm_eff)
                
                # Add to 3D surface data
                surface_data['frequencies'].append(freq)
                surface_data['flows'].append(measurement.flow)
                surface_data['normalized_eff'].append(norm_eff)
                surface_data['heads'].append(head)
                
                # Store detailed result
                result = {
                    'head': head,
                    'frequency': freq,
                    'flow': measurement.flow,
                    'power': measurement.power,
                    'proxy_value': proxy_val,
                    'true_efficiency': measurement.true_efficiency,
                    'normalized_efficiency': norm_eff,
                    'true_bep_freq': true_bep_freq,
                    'true_bep_eff': true_bep_eff
                }
                all_results.append(result)
            
            # Calculate correlation metrics
            if len(proxy_values) > 1:
                # R² between proxy and true efficiency
                r_squared_true = self._calculate_r_squared(proxy_values, true_efficiencies)
                correlation_true, p_value_true = pearsonr(proxy_values, true_efficiencies)
                
                # R² between proxy and normalized efficiency
                r_squared_norm = self._calculate_r_squared(proxy_values, normalized_efficiencies)
                correlation_norm, p_value_norm = pearsonr(proxy_values, normalized_efficiencies)
                
                correlation_analysis[head] = {
                    'r_squared_true_eff': r_squared_true,
                    'correlation_true_eff': correlation_true,
                    'p_value_true_eff': p_value_true,
                    'r_squared_normalized_eff': r_squared_norm,
                    'correlation_normalized_eff': correlation_norm,
                    'p_value_normalized_eff': p_value_norm,
                    'proxy_values': proxy_values,
                    'true_efficiencies': true_efficiencies,
                    'normalized_efficiencies': normalized_efficiencies,
                    'frequencies': frequencies_data
                }
                
                print(f"  R² (proxy vs true eff): {r_squared_true:.3f}")
                print(f"  R² (proxy vs norm eff): {r_squared_norm:.3f}")
                print(f"  Correlation (true eff): {correlation_true:.3f} (p={p_value_true:.3f})")
            
            # BEP prediction analysis
            bep_analysis[head] = self._analyze_bep_prediction(
                frequencies_data, proxy_values, true_bep_freq, true_bep_eff
            )
            
            predicted_bep = bep_analysis[head]['predicted_bep_freq']
            bep_error = abs(predicted_bep - true_bep_freq) if predicted_bep else float('inf')
            
            print(f"  Predicted BEP: {predicted_bep:.1f} Hz")
            print(f"  BEP Error: {bep_error:.1f} Hz")
        
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "csv" / "proxy_validation_detailed.csv", index=False)
        
        # Create correlation summary
        correlation_summary = []
        for head, corr_data in correlation_analysis.items():
            correlation_summary.append({
                'head': head,
                'r_squared_true_eff': corr_data['r_squared_true_eff'],
                'r_squared_normalized_eff': corr_data['r_squared_normalized_eff'],
                'correlation_true_eff': corr_data['correlation_true_eff'],
                'correlation_normalized_eff': corr_data['correlation_normalized_eff'],
                'p_value_true_eff': corr_data['p_value_true_eff']
            })
        
        correlation_df = pd.DataFrame(correlation_summary)
        correlation_df.to_csv(self.output_dir / "csv" / "proxy_correlation_summary.csv", index=False)
        
        # Create BEP analysis summary
        bep_summary = []
        for head, bep_data in bep_analysis.items():
            bep_summary.append({
                'head': head,
                'true_bep_freq': bep_data['true_bep_freq'],
                'predicted_bep_freq': bep_data['predicted_bep_freq'],
                'bep_error': bep_data['bep_error'],
                'max_proxy_value': bep_data['max_proxy_value']
            })
        
        bep_df = pd.DataFrame(bep_summary)
        bep_df.to_csv(self.output_dir / "csv" / "bep_prediction_summary.csv", index=False)
        
        # Generate visualizations
        self._create_correlation_plots(correlation_analysis)
        self._create_3d_surface_plot(surface_data)
        self._create_bep_analysis_plots(correlation_analysis, bep_analysis)
        
        # Print comprehensive summary
        self._print_validation_summary(correlation_analysis, bep_analysis)
        
        return {
            'detailed_results': all_results,
            'correlation_analysis': correlation_analysis,
            'bep_analysis': bep_analysis,
            'surface_data': surface_data,
            'test_parameters': {
                'test_heads': test_heads,
                'freq_range': freq_range,
                'n_points': n_points
            }
        }
    
    def _calculate_r_squared(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate R² coefficient of determination properly"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        # Convert to numpy arrays
        x = np.array(x_values)
        y = np.array(y_values)
        
        # Remove any NaN or infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return 0.0
        
        # Calculate R² using the proper formula: 1 - (SS_res / SS_tot)
        # First, fit a linear regression
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        
        # Calculate sum of squares
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        # R² calculation
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Ensure R² is between 0 and 1
        return max(0.0, min(1.0, r_squared))
    
    def _analyze_bep_prediction(self, frequencies: List[float], proxy_values: List[float], 
                               true_bep_freq: float, true_bep_eff: float) -> Dict[str, Any]:
        """Analyze BEP prediction capability of proxy function"""
        
        if not proxy_values or len(proxy_values) < 3:
            return {
                'predicted_bep_freq': None,
                'bep_error': float('inf'),
                'max_proxy_value': 0,
                'true_bep_freq': true_bep_freq
            }
        
        # Find frequency with maximum proxy value
        max_idx = np.argmax(proxy_values)
        predicted_bep_freq = frequencies[max_idx]
        max_proxy_value = proxy_values[max_idx]
        
        # Calculate absolute error
        bep_error = abs(predicted_bep_freq - true_bep_freq)
        
        return {
            'predicted_bep_freq': predicted_bep_freq,
            'bep_error': bep_error,
            'max_proxy_value': max_proxy_value,
            'true_bep_freq': true_bep_freq,
            'true_bep_eff': true_bep_eff
        }
    
    def _create_correlation_plots(self, correlation_analysis: Dict):
        """Create correlation analysis plots"""
        
        # Plot 1: Proxy vs True Efficiency for each head
        fig, axes = plt.subplots(1, len(correlation_analysis), figsize=(15, 5))
        if len(correlation_analysis) == 1:
            axes = [axes]
        
        for i, (head, data) in enumerate(correlation_analysis.items()):
            ax = axes[i]
            
            proxy_vals = data['proxy_values']
            true_effs = data['true_efficiencies']
            r_squared = data['r_squared_true_eff']
            
            ax.scatter(proxy_vals, true_effs, alpha=0.7, s=30)
            
            # Add trend line
            z = np.polyfit(proxy_vals, true_effs, 1)
            p = np.poly1d(z)
            ax.plot(proxy_vals, p(proxy_vals), "r--", alpha=0.8)
            
            ax.set_xlabel('Q²/P Proxy Value')
            ax.set_ylabel('True Efficiency')
            ax.set_title(f'Head {head}m\nR² = {r_squared:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "proxy_correlation_analysis.png", dpi=300)
        plt.close()
        
        # Plot 2: R² comparison across heads
        plt.figure(figsize=(10, 6))
        
        heads = list(correlation_analysis.keys())
        r_squared_true = [correlation_analysis[h]['r_squared_true_eff'] for h in heads]
        r_squared_norm = [correlation_analysis[h]['r_squared_normalized_eff'] for h in heads]
        
        x = np.arange(len(heads))
        width = 0.35
        
        plt.bar(x - width/2, r_squared_true, width, label='vs True Efficiency', alpha=0.8)
        plt.bar(x + width/2, r_squared_norm, width, label='vs Normalized Efficiency', alpha=0.8)
        
        plt.xlabel('System Head (m)')
        plt.ylabel('R² Coefficient')
        plt.title('Q²/P Proxy Correlation Analysis')
        plt.xticks(x, heads)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "r_squared_comparison.png", dpi=300)
        plt.close()
    
    def _create_3d_surface_plot(self, surface_data: Dict):
        """Create 3D surface plots: frequency vs flow vs efficiency (3 separate charts)"""
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(18, 6))
        
        # Get unique heads for color coding
        heads = np.array(surface_data['heads'])
        unique_heads = sorted(list(set(heads)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_heads)))
        
        # Chart 1: True Efficiency Surface
        ax1 = fig.add_subplot(131, projection='3d')
        
        for i, head in enumerate(unique_heads):
            mask = heads == head
            frequencies = np.array(surface_data['frequencies'])[mask]
            flows = np.array(surface_data['flows'])[mask]
            norm_effs = np.array(surface_data['normalized_eff'])[mask]
            
            ax1.scatter(frequencies, flows, norm_effs, 
                       c=[colors[i]], label=f'{head}m', alpha=0.7, s=20)
            
            # Mark BEP point
            max_eff_idx = np.argmax(norm_effs)
            bep_freq = frequencies[max_eff_idx]
            bep_flow = flows[max_eff_idx] 
            bep_eff = norm_effs[max_eff_idx]
            
            ax1.scatter([bep_freq], [bep_flow], [bep_eff], 
                       c='red', marker='*', s=100, edgecolors='black', linewidth=1)
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Flow (m³/h)')
        ax1.set_zlabel('Normalized True Efficiency')
        ax1.set_title('True Efficiency Surface\n(Red stars = True BEP)')
        ax1.legend()
        
        # Chart 2: Q²/P Proxy Surface  
        ax2 = fig.add_subplot(132, projection='3d')
        
        for i, head in enumerate(unique_heads):
            mask = heads == head
            frequencies = np.array(surface_data['frequencies'])[mask]
            flows = np.array(surface_data['flows'])[mask]
            
            # Calculate Q²/P proxy values for this head
            proxy_values = []
            for j in np.where(mask)[0]:
                # Get measurement data for proxy calculation
                freq = surface_data['frequencies'][j]
                # Recreate pump for this specific point
                pump = RealisticPumpSimulator(system_head=head, noise_level=0.01)
                measurement = pump.get_measurement(freq)
                proxy = OriginalProxy()
                proxy_val = proxy.calculate(measurement)
                proxy_values.append(proxy_val)
            
            # Normalize proxy values
            if len(proxy_values) > 0:
                proxy_norm = (np.array(proxy_values) - np.min(proxy_values)) / (np.max(proxy_values) - np.min(proxy_values))
                
                ax2.scatter(frequencies, flows, proxy_norm, 
                           c=[colors[i]], label=f'{head}m', alpha=0.7, s=20)
                
                # Mark proxy BEP (max proxy value)
                proxy_bep_idx = np.argmax(proxy_values)
                bep_freq = frequencies[proxy_bep_idx]
                bep_flow = flows[proxy_bep_idx]
                bep_proxy = proxy_norm[proxy_bep_idx]
                
                ax2.scatter([bep_freq], [bep_flow], [bep_proxy], 
                           c='red', marker='*', s=100, edgecolors='black', linewidth=1)
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Flow (m³/h)')
        ax2.set_zlabel('Normalized Q²/P Proxy')
        ax2.set_title('Q²/P Proxy Surface\n(Red stars = Predicted BEP)')
        ax2.legend()
        
        # Chart 3: Overlay Comparison
        ax3 = fig.add_subplot(133, projection='3d')
        
        for i, head in enumerate(unique_heads):
            mask = heads == head
            frequencies = np.array(surface_data['frequencies'])[mask]
            flows = np.array(surface_data['flows'])[mask]
            norm_effs = np.array(surface_data['normalized_eff'])[mask]
            
            # True efficiency (solid)
            ax3.scatter(frequencies, flows, norm_effs, 
                       c=[colors[i]], alpha=0.8, s=25, label=f'{head}m True' if i == 0 else "")
            
            # Proxy values (hollow)
            proxy_values = []
            for j in np.where(mask)[0]:
                freq = surface_data['frequencies'][j]
                pump = RealisticPumpSimulator(system_head=head, noise_level=0.01)
                measurement = pump.get_measurement(freq)
                proxy = OriginalProxy()
                proxy_val = proxy.calculate(measurement)
                proxy_values.append(proxy_val)
            
            if len(proxy_values) > 0:
                proxy_norm = (np.array(proxy_values) - np.min(proxy_values)) / (np.max(proxy_values) - np.min(proxy_values))
                ax3.scatter(frequencies, flows, proxy_norm, 
                           c=[colors[i]], marker='^', alpha=0.6, s=20)
        
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Flow (m³/h)')
        ax3.set_zlabel('Normalized Value')
        ax3.set_title('True vs Proxy Comparison\n(Circles=True, Triangles=Proxy)')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "3d_surface_plots.png", dpi=300)
        plt.close()
    
    def _create_bep_analysis_plots(self, correlation_analysis: Dict, bep_analysis: Dict):
        """Create enhanced BEP prediction analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BEP Prediction Analysis: Q²/P Proxy Performance', fontsize=16)
        
        heads = list(bep_analysis.keys())
        true_beps = [bep_analysis[h]['true_bep_freq'] for h in heads]
        predicted_beps = [bep_analysis[h]['predicted_bep_freq'] for h in heads]
        errors = [bep_analysis[h]['bep_error'] for h in heads]
        
        # Plot 1: BEP prediction accuracy with error bars
        ax1 = axes[0, 0]
        
        ax1.plot(heads, true_beps, 'bo-', label='True BEP', linewidth=3, markersize=8)
        ax1.plot(heads, predicted_beps, 'rs-', label='Q²/P Predicted BEP', linewidth=3, markersize=8)
        
        # Fill area between true and predicted
        ax1.fill_between(heads, true_beps, predicted_beps, alpha=0.3, color='orange')
        
        # Add error annotations
        for i, (head, error) in enumerate(zip(heads, errors)):
            ax1.annotate(f'{error:.1f}Hz', 
                        xy=(head, predicted_beps[i]), 
                        xytext=(head, predicted_beps[i] + 2),
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        ax1.set_xlabel('System Head (m)')
        ax1.set_ylabel('BEP Frequency (Hz)')
        ax1.set_title('(a) True vs Predicted BEP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error magnitude with trend analysis
        ax2 = axes[0, 1]
        
        colors = ['green' if e <= 2.0 else 'orange' if e <= 5.0 else 'red' for e in errors]
        bars = ax2.bar(heads, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add trend line
        z = np.polyfit(heads, errors, 1)
        p = np.poly1d(z)
        trend_line = p(heads)
        ax2.plot(heads, trend_line, 'k--', linewidth=2, alpha=0.7, label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
        
        # Add threshold lines
        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Success Target')
        ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Acceptable Limit')
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{error:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('System Head (m)')
        ax2.set_ylabel('Absolute BEP Error (Hz)')
        ax2.set_title('(b) Error Analysis with Trend')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Error vs R² correlation scatter
        ax3 = axes[1, 0]
        
        r_squares = [correlation_analysis[h]['r_squared_true_eff'] for h in heads]
        
        # Scatter plot
        for i, (head, error, r2) in enumerate(zip(heads, errors, r_squares)):
            color = 'green' if error <= 2.0 else 'orange' if error <= 5.0 else 'red'
            ax3.scatter(r2, error, c=color, s=100, alpha=0.7, edgecolors='black')
            ax3.annotate(f'{head}m', xy=(r2, error), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        # Add trend line
        if len(r_squares) > 1:
            z = np.polyfit(r_squares, errors, 1)
            p = np.poly1d(z)
            r2_range = np.linspace(min(r_squares), max(r_squares), 100)
            ax3.plot(r2_range, p(r2_range), 'k--', linewidth=2, alpha=0.7)
        
        ax3.set_xlabel('R² (Proxy vs True Efficiency)')
        ax3.set_ylabel('BEP Prediction Error (Hz)')
        ax3.set_title('(c) R² vs Error Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance quadrant analysis
        ax4 = axes[1, 1]
        
        # Create quadrant plot: R² vs Error
        for i, (head, error, r2) in enumerate(zip(heads, errors, r_squares)):
            color = 'green' if error <= 2.0 else 'orange' if error <= 5.0 else 'red'
            marker_size = 200 - (error * 20)  # Smaller markers for larger errors
            
            ax4.scatter(r2, error, c=color, s=max(50, marker_size), alpha=0.7, 
                       edgecolors='black', linewidth=1)
            ax4.annotate(f'{head}m', xy=(r2, error), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add quadrant lines
        r2_median = np.median(r_squares)
        error_median = np.median(errors)
        
        ax4.axvline(x=r2_median, color='gray', linestyle=':', alpha=0.7)
        ax4.axhline(y=error_median, color='gray', linestyle=':', alpha=0.7)
        
        # Add quadrant labels
        ax4.text(0.95, 0.95, 'Poor R²\nHigh Error', transform=ax4.transAxes, 
                ha='right', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax4.text(0.95, 0.05, 'Poor R²\nLow Error', transform=ax4.transAxes, 
                ha='right', va='bottom', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        ax4.text(0.05, 0.95, 'Good R²\nHigh Error', transform=ax4.transAxes, 
                ha='left', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        ax4.text(0.05, 0.05, 'Good R²\nLow Error', transform=ax4.transAxes, 
                ha='left', va='bottom', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax4.set_xlabel('R² (Correlation Quality)')
        ax4.set_ylabel('BEP Error (Hz)')
        ax4.set_title('(d) Performance Quadrant Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "enhanced_bep_analysis.png", dpi=300)
        plt.close()
    
    def _print_validation_summary(self, correlation_analysis: Dict, bep_analysis: Dict):
        """Print comprehensive validation summary"""
        
        print("\n" + "="*60)
        print("PROXY VALIDATION SUMMARY")
        print("="*60)
        
        # R² Analysis
        print(f"\nR² CORRELATION ANALYSIS:")
        print("-" * 25)
        print(f"{'Head (m)':<8} {'R² (True)':<10} {'R² (Norm)':<10} {'Correlation':<12}")
        print("-" * 45)
        
        r_squared_values = []
        for head, data in correlation_analysis.items():
            r2_true = data['r_squared_true_eff']
            r2_norm = data['r_squared_normalized_eff']
            corr = data['correlation_true_eff']
            
            print(f"{head:<8} {r2_true:<10.3f} {r2_norm:<10.3f} {corr:<12.3f}")
            r_squared_values.append(r2_true)
        
        mean_r_squared = np.mean(r_squared_values)
        print(f"\nMean R² (vs True Efficiency): {mean_r_squared:.3f}")
        
        # BEP Prediction Analysis
        print(f"\nBEP PREDICTION ANALYSIS:")
        print("-" * 25)
        print(f"{'Head (m)':<8} {'True BEP':<10} {'Predicted':<10} {'Error (Hz)':<10}")
        print("-" * 40)
        
        errors = []
        for head, data in bep_analysis.items():
            true_bep = data['true_bep_freq']
            pred_bep = data['predicted_bep_freq']
            error = data['bep_error']
            
            print(f"{head:<8} {true_bep:<10.1f} {pred_bep:<10.1f} {error:<10.1f}")
            errors.append(error)
        
        # Summary Statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        success_rate = sum(1 for e in errors if e <= 2.0) / len(errors) * 100
        
        print(f"\nSUMMARY STATISTICS:")
        print("-" * 20)
        print(f"Mean BEP Error: {mean_error:.2f} ± {std_error:.2f} Hz")
        print(f"Maximum Error: {max_error:.2f} Hz")
        print(f"Success Rate (≤2Hz): {success_rate:.0f}%")
        print(f"Mean R²: {mean_r_squared:.3f}")
        
        # Overall Assessment
        print(f"\nOVERALL ASSESSMENT:")
        print("-" * 20)
        if mean_r_squared > 0.8 and success_rate > 80:
            assessment = "EXCELLENT - Strong correlation and high BEP accuracy"
        elif mean_r_squared > 0.6 and success_rate > 60:
            assessment = "GOOD - Reasonable correlation and acceptable BEP accuracy"
        elif mean_r_squared > 0.4:
            assessment = "MODERATE - Some correlation but limited BEP accuracy"
        else:
            assessment = "POOR - Weak correlation and poor BEP prediction"
        
        print(assessment)
        print(f"\nRecommendation: {'Deploy with confidence' if mean_r_squared > 0.7 else 'Consider improvements or alternative proxies'}")


def main():
    """Main function to run proxy validation experiment"""
    
    print("Q²/P Proxy Function Validation Experiment")
    print("=" * 45)
    print("Analyzing correlation with true BEP and efficiency")
    
    # Initialize experiment
    experiment = ProxyValidationExperiment(output_dir="results")
    
    # Run comprehensive validation
    results = experiment.run_validation_experiment(
        test_heads=[20, 30, 40],  # Low-head pump range
        freq_range=(25, 60),     # Wide frequency sweep
        n_points=50              # High resolution
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("Generated files:")
    print("• results/csv/proxy_validation_detailed.csv")
    print("• results/csv/proxy_correlation_summary.csv") 
    print("• results/csv/bep_prediction_summary.csv")
    print("• results/plots/proxy_correlation_analysis.png")
    print("• results/plots/r_squared_comparison.png")
    print("• results/plots/3d_surface_plot.png")
    print("• results/plots/bep_prediction_accuracy.png")
    print("• results/plots/absolute_error_analysis.png")
    print("\nReady for paper analysis!")


if __name__ == "__main__":
    main()