# ==============================================================================
# Head Sensitivity Analysis: Q²/P vs Q/√P Performance Investigation
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes
from src.pump_model import RealisticPumpSimulator
from src.proxy_functions import NormalizedProxy, OriginalProxy, LinearProxy
from src.utils import setup_publication_style

class HeadSensitivityAnalyzer:
    """Analyze how Q²/P vs Q/√P proxies perform at different head conditions"""
    
    def __init__(self):
        setup_publication_style()
        
    def analyze_head_sensitivity(self, 
                                test_heads: List[float] = [15, 20, 25, 30, 35, 40, 45, 50, 55],
                                freq_range: Tuple[float, float] = (25, 60),
                                n_points: int = 40) -> Dict:
        """
        Comprehensive analysis of proxy performance vs head conditions
        Focus: Why Q²/P vs Q/√P behave differently at low/high heads
        """
        
        print(f"Analyzing head sensitivity: {min(test_heads)}-{max(test_heads)}m range")
        print("Theory: Q²/P should be more head-sensitive than Q/√P")
        
        # Define the three proxy methods for comprehensive comparison
        proxy_methods = {
            'Linear Q/P×PF': LinearProxy(),
            'Original Q²/P×PF': OriginalProxy(), 
            'Normalized Q/√P×PF': NormalizedProxy()
        }
        
        results = {}
        
        for head in test_heads:
            print(f"  Testing head {head}m...")
            
            # Setup pump
            pump = RealisticPumpSimulator(system_head=head, noise_level=0.015)
            frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
            
            # Collect measurements
            flows = []
            powers = []
            true_efficiencies = []
            head_rises = []  # Actual head rise from pump
            
            for freq in frequencies:
                measurement = pump.get_measurement(freq)
                flows.append(measurement.flow)
                powers.append(measurement.power)
                true_efficiencies.append(measurement.true_efficiency)
                # Calculate head rise (total head - system head)
                head_rise = measurement.true_head - head if measurement.true_head else 0
                head_rises.append(head_rise)
            
            # Calculate proxy values
            proxy_values = {}
            for name, proxy_func in proxy_methods.items():
                values = []
                for i in range(len(flows)):
                    # Create measurement object for proxy calculation
                    from src.pump_model import PumpMeasurement
                    import time
                    meas = PumpMeasurement(
                        timestamp=time.time(),
                        frequency=frequencies[i],
                        flow=flows[i],
                        power=powers[i],
                        voltage=400.0,
                        current=powers[i] * 1000 / (400 * np.sqrt(3) * 0.85),
                        power_factor=0.85,
                        true_efficiency=true_efficiencies[i]
                    )
                    proxy_val = proxy_func.calculate(meas)
                    values.append(proxy_val)
                proxy_values[name] = values
            
            # Get true BEP
            true_bep_freq, true_bep_eff = pump.get_true_bep()
            true_bep_measurement = pump.get_measurement(true_bep_freq)
            true_bep_flow = true_bep_measurement.flow
            
            # Store results
            results[head] = {
                'flows': flows,
                'powers': powers,
                'true_efficiencies': true_efficiencies,
                'head_rises': head_rises,
                'proxy_values': proxy_values,
                'frequencies': frequencies,
                'true_bep': {
                    'frequency': true_bep_freq,
                    'flow': true_bep_flow,
                    'efficiency': true_bep_eff,
                    'power': true_bep_measurement.power
                },
                'system_head': head
            }
        
        return results
    
    def calculate_head_dependent_performance(self, results: Dict) -> Dict:
        """
        Calculate performance metrics with focus on head dependency
        """
        
        test_heads = sorted(results.keys())
        proxy_names = list(results[test_heads[0]]['proxy_values'].keys())
        
        performance = {}
        
        for proxy_name in proxy_names:
            performance[proxy_name] = {
                'head_conditions': [],
                'bep_flow_errors': [],
                'bep_freq_errors': [],
                'correlations': [],
                'proxy_bep_flows': [],
                'proxy_bep_freqs': [],
                'true_bep_flows': [],
                'head_bias_analysis': {},
                'theoretical_behavior': {}
            }
            
            for head in test_heads:
                data = results[head]
                proxy_vals = data['proxy_values'][proxy_name]
                flows = data['flows']
                frequencies = data['frequencies']
                true_effs = data['true_efficiencies']
                
                # Find proxy BEP
                max_idx = np.argmax(proxy_vals)
                proxy_bep_flow = flows[max_idx]
                proxy_bep_freq = frequencies[max_idx]
                
                # True BEP
                true_bep_flow = data['true_bep']['flow']
                true_bep_freq = data['true_bep']['frequency']
                
                # Calculate errors
                flow_error = abs(proxy_bep_flow - true_bep_flow)
                freq_error = abs(proxy_bep_freq - true_bep_freq)
                
                # Correlation
                correlation = np.corrcoef(proxy_vals, true_effs)[0, 1]
                
                # Store data
                performance[proxy_name]['head_conditions'].append(head)
                performance[proxy_name]['bep_flow_errors'].append(flow_error)
                performance[proxy_name]['bep_freq_errors'].append(freq_error)
                performance[proxy_name]['correlations'].append(correlation)
                performance[proxy_name]['proxy_bep_flows'].append(proxy_bep_flow)
                performance[proxy_name]['proxy_bep_freqs'].append(proxy_bep_freq)
                performance[proxy_name]['true_bep_flows'].append(true_bep_flow)
            
            # Analyze head bias patterns
            heads = performance[proxy_name]['head_conditions']
            errors = performance[proxy_name]['bep_flow_errors']
            
            # Low head performance (15-30m)
            low_head_mask = np.array(heads) <= 30
            low_head_errors = np.array(errors)[low_head_mask]
            
            # High head performance (40-55m)
            high_head_mask = np.array(heads) >= 40
            high_head_errors = np.array(errors)[high_head_mask]
            
            performance[proxy_name]['head_bias_analysis'] = {
                'low_head_mean_error': np.mean(low_head_errors) if len(low_head_errors) > 0 else 0,
                'high_head_mean_error': np.mean(high_head_errors) if len(high_head_errors) > 0 else 0,
                'low_head_std': np.std(low_head_errors) if len(low_head_errors) > 0 else 0,
                'high_head_std': np.std(high_head_errors) if len(high_head_errors) > 0 else 0,
                'bias_magnitude': np.mean(high_head_errors) - np.mean(low_head_errors) if len(low_head_errors) > 0 and len(high_head_errors) > 0 else 0,
                'overall_sensitivity': np.std(errors)
            }
            
            # Theoretical analysis - Updated for three proxies
            if 'Q²/P' in proxy_name:
                theory = "Q²/P increases quadratically with flow, strongly amplifying head-induced flow changes. Higher heads → different Q-H relationship → highest sensitivity of all three methods."
                expected_bias = "HIGH POSITIVE (worst at high heads)"
            elif 'Q/√P' in proxy_name:
                theory = "Q/√P increases linearly with flow but compensates via √P term, making it most robust to head variations. Square root normalization provides optimal head compensation."
                expected_bias = "MINIMAL (most robust across heads)"
            else:  # Q/P - Linear proxy
                theory = "Q/P increases linearly with flow, moderately sensitive to head changes. Intermediate behavior between Q²/P and Q/√P. No compensation mechanism."
                expected_bias = "MODERATE POSITIVE (moderate sensitivity)"
            
            performance[proxy_name]['theoretical_behavior'] = {
                'theory': theory,
                'expected_bias': expected_bias,
                'actual_bias': 'HIGH' if performance[proxy_name]['head_bias_analysis']['bias_magnitude'] > 8 else 'MODERATE' if performance[proxy_name]['head_bias_analysis']['bias_magnitude'] > 3 else 'MINIMAL'
            }
        
        return performance
    
    def plot_head_sensitivity_comparison(self, results: Dict, performance: Dict):
        """
        Create comprehensive head sensitivity comparison plot
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Head Sensitivity Analysis: Q/P×PF vs Q²/P×PF vs Q/√P×PF', fontsize=16)
        
        test_heads = sorted(results.keys())
        proxy_names = list(performance.keys())
        
        # Colors for the three methods
        colors = {
            'Linear Q/P×PF': '#28B463',        # Green  
            'Original Q²/P×PF': '#E74C3C',     # Red
            'Normalized Q/√P×PF': '#2E86C1'    # Blue
        }
        
        # Plot 1: Flow vs Efficiency curves at different heads
        ax1 = axes[0, 0]
        
        # Show characteristic curves for low, medium, high head
        representative_heads = [20, 35, 50]
        head_colors = ['green', 'blue', 'red']
        
        for i, head in enumerate(representative_heads):
            if head in results:
                data = results[head]
                flows = data['flows']
                true_effs = data['true_efficiencies']
                true_bep_flow = data['true_bep']['flow']
                true_bep_eff = data['true_bep']['efficiency']
                
                ax1.plot(flows, true_effs, 'o-', color=head_colors[i], linewidth=2, 
                        markersize=3, label=f'{head}m head', alpha=0.8)
                ax1.plot(true_bep_flow, true_bep_eff, 'o', color=head_colors[i], 
                        markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        ax1.set_xlabel('Flow Rate (m³/h)')
        ax1.set_ylabel('True Efficiency')
        ax1.set_title('(a) Efficiency Curves: Head Effect')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: BEP Detection Error vs Head
        ax2 = axes[0, 1]
        
        for proxy_name in proxy_names:
            heads = performance[proxy_name]['head_conditions']
            errors = performance[proxy_name]['bep_flow_errors']
            ax2.plot(heads, errors, 'o-', color=colors[proxy_name], linewidth=2, 
                    markersize=6, label=proxy_name.split()[0])
        
        ax2.set_xlabel('System Head (m)')
        ax2.set_ylabel('BEP Flow Error (m³/h)')
        ax2.set_title('(b) BEP Error vs Head')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add trend lines
        for proxy_name in proxy_names:
            heads = performance[proxy_name]['head_conditions']
            errors = performance[proxy_name]['bep_flow_errors']
            z = np.polyfit(heads, errors, 1)
            p = np.poly1d(z)
            ax2.plot(heads, p(heads), '--', color=colors[proxy_name], alpha=0.5)
        
        # Plot 3: Correlation Quality vs Head
        ax3 = axes[0, 2]
        
        for proxy_name in proxy_names:
            heads = performance[proxy_name]['head_conditions']
            correlations = performance[proxy_name]['correlations']
            ax3.plot(heads, correlations, 'o-', color=colors[proxy_name], linewidth=2, 
                    markersize=6, label=proxy_name.split()[0])
        
        ax3.set_xlabel('System Head (m)')
        ax3.set_ylabel('Correlation with True Efficiency')
        ax3.set_title('(c) Proxy Quality vs Head')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Low vs High Head Performance
        ax4 = axes[1, 0]
        
        categories = ['Low Head\n(≤30m)', 'High Head\n(≥40m)']
        x_pos = np.arange(len(categories))
        width = 0.35
        
        for i, proxy_name in enumerate(proxy_names):
            bias_data = performance[proxy_name]['head_bias_analysis']
            values = [bias_data['low_head_mean_error'], bias_data['high_head_mean_error']]
            errors = [bias_data['low_head_std'], bias_data['high_head_std']]
            
            ax4.bar(x_pos + i*width, values, width, yerr=errors, capsize=5,
                   label=proxy_name.split()[0], color=colors[proxy_name], alpha=0.7)
        
        ax4.set_xlabel('Head Condition')
        ax4.set_ylabel('Mean BEP Error (m³/h)')
        ax4.set_title('(d) Low vs High Head Performance')
        ax4.set_xticks(x_pos + width/2)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Head Sensitivity Comparison
        ax5 = axes[1, 1]
        
        sensitivity_metrics = []
        sensitivity_labels = []
        sensitivity_colors = []
        
        for proxy_name in proxy_names:
            bias_data = performance[proxy_name]['head_bias_analysis']
            sensitivity_metrics.append(bias_data['overall_sensitivity'])
            sensitivity_labels.append(proxy_name.split()[0])
            sensitivity_colors.append(colors[proxy_name])
        
        bars = ax5.bar(sensitivity_labels, sensitivity_metrics, 
                      color=sensitivity_colors, alpha=0.7)
        ax5.set_ylabel('Head Sensitivity (m³/h std)')
        ax5.set_title('(e) Overall Head Sensitivity\n(Lower = More Robust)')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, sensitivity_metrics):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Theoretical vs Actual Analysis
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create theoretical analysis summary
        summary_text = "THEORETICAL vs ACTUAL BEHAVIOR\n"
        summary_text += "=" * 35 + "\n\n"
        
        for proxy_name in proxy_names:
            if 'Linear Q/P' in proxy_name:
                short_name = "Q/P"
                math_form = "Q/P"
            elif 'Original Q²/P' in proxy_name:
                short_name = "Q²/P"
                math_form = "Q²/P"
            else:
                short_name = "Q/√P"
                math_form = "Q/√P"
            
            theory_data = performance[proxy_name]['theoretical_behavior']
            bias_data = performance[proxy_name]['head_bias_analysis']
            
            summary_text += f"{short_name} ({math_form}):\n"
            summary_text += f"Expected: {theory_data['expected_bias'].split()[0]}\n"
            summary_text += f"Actual: {theory_data['actual_bias']}\n"
            summary_text += f"Bias: {bias_data['bias_magnitude']:+.1f} m³/h\n"
            summary_text += f"Sensitivity: {bias_data['overall_sensitivity']:.1f} m³/h\n\n"
        
        # Add theoretical explanation
        summary_text += "THEORETICAL PROGRESSION:\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "Q/P: Linear flow dependency\n"
        summary_text += "→ Baseline head sensitivity\n\n"
        summary_text += "Q²/P: Quadratic flow dependency\n"
        summary_text += "→ Amplified head sensitivity\n\n"
        summary_text += "Q/√P: Linear flow, √P normalization\n"
        summary_text += "→ Optimal head compensation\n\n"
        
        # Conclusion
        best_method = min(proxy_names, key=lambda x: performance[x]['head_bias_analysis']['overall_sensitivity'])
        worst_method = max(proxy_names, key=lambda x: performance[x]['head_bias_analysis']['overall_sensitivity'])
        
        summary_text += f"RANKING (Most to Least Robust):\n"
        # Sort by sensitivity (lower = better)
        sorted_methods = sorted(proxy_names, key=lambda x: performance[x]['head_bias_analysis']['overall_sensitivity'])
        for i, method in enumerate(sorted_methods):
            if 'Linear Q/P' in method:
                name = "Q/P"
            elif 'Original Q²/P' in method:
                name = "Q²/P"  
            else:
                name = "Q/√P"
            sensitivity = performance[method]['head_bias_analysis']['overall_sensitivity']
            summary_text += f"{i+1}. {name} ({sensitivity:.1f} m³/h)\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('head_sensitivity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Run head sensitivity analysis"""
    
    print("HEAD SENSITIVITY ANALYSIS: Q/P×PF vs Q²/P×PF vs Q/√P×PF")
    print("=" * 60)
    print("Research Question: How do these three proxies perform at different heads?")
    print("Theory: Q/P < Q/√P < Q²/P in terms of head sensitivity")
    
    analyzer = HeadSensitivityAnalyzer()
    
    # Comprehensive head range analysis
    print("\n1. Analyzing performance across head range...")
    results = analyzer.analyze_head_sensitivity(
        test_heads=[15, 20, 25, 30, 35, 40, 45, 50, 55]
    )
    
    # Calculate head-dependent performance metrics
    print("\n2. Calculating head-dependent performance metrics...")
    performance = analyzer.calculate_head_dependent_performance(results)
    
    # Create comprehensive comparison plots
    print("\n3. Creating head sensitivity comparison plots...")
    fig = analyzer.plot_head_sensitivity_comparison(results, performance)
    
    # Print quantitative results
    print("\n" + "="*60)
    print("HEAD SENSITIVITY ANALYSIS RESULTS")
    print("="*60)
    
    for proxy_name in performance.keys():
        if 'Linear Q/P' in proxy_name:
            short_name = "Linear Q/P"
            math_form = "Q/P×PF"
        elif 'Original Q²/P' in proxy_name:
            short_name = "Original Q²/P"
            math_form = "Q²/P×PF"
        else:
            short_name = "Normalized Q/√P"
            math_form = "Q/√P×PF"
        
        bias_data = performance[proxy_name]['head_bias_analysis']
        theory_data = performance[proxy_name]['theoretical_behavior']
        
        print(f"\n{short_name} ({math_form}):")
        print(f"  Low Head Error (≤30m):  {bias_data['low_head_mean_error']:.1f} ± {bias_data['low_head_std']:.1f} m³/h")
        print(f"  High Head Error (≥40m): {bias_data['high_head_mean_error']:.1f} ± {bias_data['high_head_std']:.1f} m³/h")
        print(f"  Head Bias:              {bias_data['bias_magnitude']:+.1f} m³/h")
        print(f"  Overall Sensitivity:    {bias_data['overall_sensitivity']:.1f} m³/h")
        print(f"  Expected Behavior:      {theory_data['expected_bias'].split()[0]}")
        print(f"  Actual Behavior:        {theory_data['actual_bias']}")
        print(f"  Theory Match:           {'✓ YES' if theory_data['expected_bias'].split()[0] in theory_data['actual_bias'] else '✗ NO'}")
    
    # Key findings
    print(f"\nKEY FINDINGS:")
    
    # Find most/least sensitive
    sensitivities = {name: performance[name]['head_bias_analysis']['overall_sensitivity'] 
                    for name in performance.keys()}
    most_robust = min(sensitivities.keys(), key=lambda x: sensitivities[x])
    most_sensitive = max(sensitivities.keys(), key=lambda x: sensitivities[x])
    
    # Convert names for display
    def get_display_name(name):
        if 'Linear Q/P' in name:
            return "Q/P"
        elif 'Original Q²/P' in name:
            return "Q²/P"
        else:
            return "Q/√P"
    
    print(f"• Most Head-Robust:    {get_display_name(most_robust)} ({sensitivities[most_robust]:.1f} m³/h sensitivity)")
    print(f"• Most Head-Sensitive: {get_display_name(most_sensitive)} ({sensitivities[most_sensitive]:.1f} m³/h sensitivity)")
    
    # Bias analysis for all three
    qp_bias = performance['Linear Q/P×PF']['head_bias_analysis']['bias_magnitude']
    q2p_bias = performance['Original Q²/P×PF']['head_bias_analysis']['bias_magnitude']
    qsqrtp_bias = performance['Normalized Q/√P×PF']['head_bias_analysis']['bias_magnitude']
    
    print(f"• Q/P Head Bias:       {qp_bias:+.1f} m³/h (high vs low head)")
    print(f"• Q²/P Head Bias:      {q2p_bias:+.1f} m³/h (high vs low head)")
    print(f"• Q/√P Head Bias:      {qsqrtp_bias:+.1f} m³/h (high vs low head)")
    
    # Ranking by sensitivity
    sorted_methods = sorted(performance.keys(), key=lambda x: performance[x]['head_bias_analysis']['overall_sensitivity'])
    print(f"\nHEAD ROBUSTNESS RANKING (Best → Worst):")
    for i, method in enumerate(sorted_methods):
        name = get_display_name(method)
        sensitivity = performance[method]['head_bias_analysis']['overall_sensitivity']
        bias = performance[method]['head_bias_analysis']['bias_magnitude']
        print(f"  {i+1}. {name}: {sensitivity:.1f} m³/h sensitivity, {bias:+.1f} m³/h bias")
    
    print(f"\nCONCLUSION FOR PAPER:")
    best_name = get_display_name(most_robust)
    worst_name = get_display_name(most_sensitive)
    
    print(f"✓ {best_name} shows superior head robustness ({sensitivities[most_robust]:.1f} m³/h sensitivity)")
    print(f"✓ Theory validation: {worst_name} shows highest sensitivity ({sensitivities[most_sensitive]:.1f} m³/h)")
    
    # Theoretical progression validation
    qp_sens = sensitivities['Linear Q/P×PF']
    q2p_sens = sensitivities['Original Q²/P×PF']  
    qsqrtp_sens = sensitivities['Normalized Q/√P×PF']
    
    if qsqrtp_sens < qp_sens < q2p_sens:
        print(f"✓ Theoretical progression confirmed: Q/√P < Q/P < Q²/P sensitivity")
    elif qsqrtp_sens < q2p_sens < qp_sens:
        print(f"✓ Partial confirmation: Q/√P best, but Q/P vs Q²/P order unexpected")
    else:
        print(f"⚠ Unexpected sensitivity ranking - requires further investigation")
    
    print(f"\nGenerated: head_sensitivity_comparison.png")
    
    return results, performance

if __name__ == "__main__":
    results, performance = main()