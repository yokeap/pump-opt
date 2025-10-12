# ==============================================================================
# Static Head BEP Detection Test
# Tests ESC optimizer with firstOrderProxy under constant head (35m)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import json
from pathlib import Path
import sys
from scipy.signal import savgol_filter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions import firstOrderProxy
from src.optimizers import ExtremumSeekingControl, TPEOptimizer

rcParams.update({
    # --- Font and text ---
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 15,  # base font size for papers (10–11pt is typical)
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'mathtext.fontset': 'cm',   # Use Computer Modern math font (LaTeX-style)
    'axes.unicode_minus': False,

    # --- Lines and markers ---
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,

    # --- Figure layout ---
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',

    # --- Colors ---
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']),

    # --- Legend style ---
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,

    # --- Grid style (disabled for Origin-like look) ---
    'axes.grid': False,
})

def find_true_bep_at_constant_head(pump, target_head=35.0, freq_range=(30, 65, 200)):
    """Find true BEP by scanning frequencies at constant head"""
    frequencies = np.linspace(*freq_range)
    
    flows, efficiencies, valid_freqs = [], [], []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.5)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        eff = pump._calculate_pump_efficiency(flow, freq)
        
        flows.append(flow)
        efficiencies.append(eff)
        valid_freqs.append(freq)
    
    flows = np.array(flows)
    efficiencies = np.array(efficiencies)
    valid_freqs = np.array(valid_freqs)
    
    bep_idx = np.argmax(efficiencies)
    
    return {
        'flow': flows[bep_idx],
        'efficiency': efficiencies[bep_idx],
        'frequency': valid_freqs[bep_idx],
        'all_flows': flows,
        'all_efficiencies': efficiencies,
        'all_frequencies': valid_freqs
    }

def run_optimizer_test(optimizer, pump, target_head=35.0, max_iterations=30, time_per_eval=5.0):
    """Run optimizer to find BEP"""
    
    results = {
        'iterations': [],
        'frequencies': [],
        'flows': [],
        'powers': [],
        'proxy_values': [],
        'true_efficiencies': [],
        'convergence': [],
        'time_minutes': []
    }
    
    best_proxy_so_far = -np.inf
    
    for iteration in range(1, max_iterations + 1):
        freq = optimizer.suggest_frequency()
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.5)
        
        if flow is None:
            continue
        
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor, true_efficiency):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
                self.true_efficiency = true_efficiency
        
        measurement = SimpleMeasurement(flow, power, pf, eff)
        proxy_value = optimizer.proxy_function.calculate(measurement)
        optimizer.update(freq, measurement)
        best_proxy_so_far = max(best_proxy_so_far, proxy_value)
        
        results['iterations'].append(iteration)
        results['frequencies'].append(freq)
        results['flows'].append(flow)
        results['powers'].append(power)
        results['proxy_values'].append(proxy_value)
        results['true_efficiencies'].append(eff)
        results['convergence'].append(best_proxy_so_far)
        results['time_minutes'].append(iteration * time_per_eval)
    
    for key in results:
        if key != 'iterations':
            results[key] = np.array(results[key])
    
    best_idx = np.argmax(results['proxy_values'])
    
    results['predicted_bep'] = {
        'flow': results['flows'][best_idx],
        'efficiency': results['true_efficiencies'][best_idx],
        'frequency': results['frequencies'][best_idx],
        'proxy_value': results['proxy_values'][best_idx],
        'iteration': results['iterations'][best_idx]
    }
    
    return results

def calculate_metrics(true_bep, predicted_bep, results):
    """Calculate performance metrics"""
    flow_error = abs(predicted_bep['flow'] - true_bep['flow'])
    relative_flow_error = (flow_error / true_bep['flow']) * 100
    eff_difference = abs(predicted_bep['efficiency'] - true_bep['efficiency']) * 100
    
    flow_tolerance = true_bep['flow'] * 0.05
    evals_to_reach_bep = len(results['flows'])
    
    for i, flow in enumerate(results['flows']):
        if abs(flow - true_bep['flow']) <= flow_tolerance:
            evals_to_reach_bep = i + 1
            break
    
    time_to_reach_bep = evals_to_reach_bep * 5.0
    
    return {
        'relative_flow_error_pct': relative_flow_error,
        'efficiency_difference_pct': eff_difference,
        'evaluations_to_bep': evals_to_reach_bep,
        'time_to_bep_minutes': time_to_reach_bep,
        'total_evaluations': len(results['flows']),
        'flow_error_abs': flow_error
    }

def create_visualizations(true_bep, results_dict, output_dir):
    """Create all required visualizations"""
    colors = {
        'ESC': '#D95319',
        'True BEP': '#77AC30'
    }

    flows = true_bep['all_flows']
    effs = true_bep['all_efficiencies']

    # Sort first
    sorted_idx = np.argsort(flows)
    flows_sorted = flows[sorted_idx]
    effs_sorted = effs[sorted_idx]

    # Apply Savitzky-Golay smoothing
    effs_smooth = savgol_filter(effs_sorted, window_length=11, polyorder=2)  # adjust window_length
    
    # Chart 1: True vs Predicted Efficiency Curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.plot(true_bep['all_flows'], true_bep['all_efficiencies'] * 100,
    #          'k-', linewidth=1.5, label='True Efficiency', zorder=1)
    ax1.plot(flows_sorted, effs_smooth * 100, 'k-', linewidth=2.0, label='True Efficiency', zorder=1)
    ax1.plot(true_bep['flow'], true_bep['efficiency'] * 100,
             marker='s', color=colors['True BEP'], markersize=10,
             markeredgecolor='black', markeredgewidth=1.5,
             label='True BEP', zorder=5)
    
    for opt_name, results in results_dict.items():
        flows = results['flows']
        effs = results['true_efficiencies'] * 100
        ax1.scatter(flows, effs, alpha=0.3, s=40, color=colors['ESC'], zorder=2)
        pred_bep = results['predicted_bep']
        ax1.plot(pred_bep['flow'], pred_bep['efficiency'] * 100,
                marker='v', color=colors['ESC'], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'{opt_name} Predicted BEP', zorder=4)
    
    ax1.set_xlabel('Flow Rate (m³/h)', fontweight='bold')
    ax1.set_ylabel('Efficiency (%)', fontweight='bold')
    ax1.set_title('True vs Predicted Efficiency Curve (H = 35m)', fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_curve_comparison.png'), dpi=300)
    plt.close()
    
    # Chart 2: Convergence Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for opt_name, results in results_dict.items():
        iterations = results['iterations']
        convergence = results['convergence']
        convergence_normalized = (convergence - convergence.min()) / (convergence.max() - convergence.min())
        eff_min = min(results['true_efficiencies']) * 100
        eff_max = max(results['true_efficiencies']) * 100
        convergence_eff = eff_min + convergence_normalized * (eff_max - eff_min)
        ax2.plot(iterations, convergence_eff, linewidth=2.5, marker='o', markersize=4,
                 color=colors['ESC'], label=opt_name)
    ax2.axhline(y=true_bep['efficiency'] * 100, color=colors['True BEP'],
                linestyle='--', linewidth=2, label='True BEP Efficiency')
    ax2.set_xlabel('Evaluations (Iterations)', fontweight='bold')
    ax2.set_ylabel('Best Efficiency Found (%)', fontweight='bold')
    ax2.set_title('Convergence to BEP', fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_plot.png'), dpi=300)
    plt.close()
    
    print("\n✓ Visualizations saved to:", output_dir)

def main():
    print("="*80)
    print("STATIC HEAD BEP DETECTION TEST")
    print("Testing ESC with firstOrderProxy at H = 35m")
    print("="*80)
    
    output_dir = Path('results/static/esc/30m')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pump = CommercialPumpSimulator(system_head=30.0, noise_level=0.00)
    proxy = firstOrderProxy()
    
    target_head = 30.0
    max_iterations = 30
    
    print("\n" + "="*80)
    print("FINDING TRUE BEP AT H = 30m")
    print("="*80)
    true_bep = find_true_bep_at_constant_head(pump, target_head)
    
    print(f"\nTrue BEP Found:")
    print(f"  Flow:       {true_bep['flow']:.3f} m³/h")
    print(f"  Efficiency: {true_bep['efficiency']*100:.2f}%")
    print(f"  Frequency:  {true_bep['frequency']:.1f} Hz")
    
    optimizer = ExtremumSeekingControl(freq_min=30, freq_max=65, step_size=2.0, proxy_function=proxy)
    # optimizer = TPEOptimizer(freq_min=30, freq_max=65, proxy_function=proxy)
    results = run_optimizer_test(optimizer, pump, target_head, max_iterations)
    
    print(f"\nPredicted BEP:")
    print(f"  Flow:       {results['predicted_bep']['flow']:.3f} m³/h")
    print(f"  Efficiency: {results['predicted_bep']['efficiency']*100:.2f}%")
    print(f"  Frequency:  {results['predicted_bep']['frequency']:.1f} Hz")
    print(f"  Found at:   Iteration {results['predicted_bep']['iteration']}")
    
    metrics = calculate_metrics(true_bep, results['predicted_bep'], results)
    
    print(f"\nMetrics:")
    print(f"  Flow Error:           {metrics['relative_flow_error_pct']:.2f}%")
    print(f"  Efficiency Diff:      {metrics['efficiency_difference_pct']:.2f}%")
    print(f"  Evaluations to BEP:   {metrics['evaluations_to_bep']}")
    print(f"  Time to BEP:          {metrics['time_to_bep_minutes']:.1f} minutes")
    
    results_dict = {'ESC': results}
    create_visualizations(true_bep, results_dict, output_dir)
    
    summary = {
        'test_parameters': {
            'target_head': target_head,
            'max_iterations': max_iterations,
            'time_per_evaluation': 2.5,
            'frequency_range': [30, 65],
            'proxy_function': 'firstOrderProxy'
        },
        'true_bep': {
            'flow': float(true_bep['flow']),
            'efficiency': float(true_bep['efficiency']),
            'frequency': float(true_bep['frequency'])
        },
        'optimizer': {
            'predicted_bep': {
                'flow': float(results['predicted_bep']['flow']),
                'efficiency': float(results['predicted_bep']['efficiency']),
                'frequency': float(results['predicted_bep']['frequency']),
                'iteration': int(results['predicted_bep']['iteration'])
            },
            'metrics': metrics
        }
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'test_results.json'}")

if __name__ == "__main__":
    main()
