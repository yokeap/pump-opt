# ==============================================================================
# Static Head BEP Detection Test
# Tests ESC optimizer with firstOrderProxy under constant head (35m)
# FIXED: time_to_bep_minutes calculation
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
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
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
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']),
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
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

def run_optimizer_test(optimizer, pump, target_head=35.0, max_iterations=30, time_per_eval=2.5):
    """
    Run optimizer to find BEP
    
    Args:
        optimizer: Optimizer instance
        pump: Pump simulator
        target_head: Target head in meters
        max_iterations: Maximum number of iterations
        time_per_eval: Time per evaluation in MINUTES (default: 2.5 min)
    """
    
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
        # FIXED: Use time_per_eval parameter correctly
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
        'iteration': results['iterations'][best_idx],
        'time_minutes': results['time_minutes'][best_idx]  # Add time to predicted_bep
    }
    
    return results

def calculate_metrics(true_bep, predicted_bep, results, time_per_eval=2.5):
    """
    Calculate performance metrics
    
    Args:
        true_bep: True BEP dictionary
        predicted_bep: Predicted BEP dictionary
        results: Results dictionary from run_optimizer_test
        time_per_eval: Time per evaluation in MINUTES (must match run_optimizer_test)
    """
    flow_error = abs(predicted_bep['flow'] - true_bep['flow'])
    relative_flow_error = (flow_error / true_bep['flow']) * 100
    eff_difference = abs(predicted_bep['efficiency'] - true_bep['efficiency']) * 100
    
    # Define convergence criterion: within 5% of true BEP flow
    flow_tolerance = true_bep['flow'] * 0.05
    evals_to_reach_bep = len(results['flows'])  # Default: didn't reach
    time_to_reach_bep = evals_to_reach_bep * time_per_eval
    
    # Find first time we reached within tolerance
    for i, flow in enumerate(results['flows']):
        if abs(flow - true_bep['flow']) <= flow_tolerance:
            evals_to_reach_bep = i + 1
            # FIXED: Calculate time correctly using time_per_eval parameter
            time_to_reach_bep = evals_to_reach_bep * time_per_eval
            break
    
    return {
        'relative_flow_error_pct': relative_flow_error,
        'efficiency_difference_pct': eff_difference,
        'evaluations_to_bep': evals_to_reach_bep,
        'time_to_bep_minutes': time_to_reach_bep,
        'total_evaluations': len(results['flows']),
        'total_time_minutes': len(results['flows']) * time_per_eval,
        'flow_error_abs': flow_error,
        'converged': evals_to_reach_bep < len(results['flows'])
    }

def create_visualizations(true_bep, results_dict, target_head,output_dir):
    """Create all required visualizations"""
    colors = {
        'ESC': '#D95319',
        'TPE': '#0072BD',
        'True BEP': '#77AC30'
    }

    flows = true_bep['all_flows']
    effs = true_bep['all_efficiencies']

    # Sort first
    sorted_idx = np.argsort(flows)
    flows_sorted = flows[sorted_idx]
    effs_sorted = effs[sorted_idx]

    # Apply Savitzky-Golay smoothing
    if len(effs_sorted) >= 11:
        effs_smooth = savgol_filter(effs_sorted, window_length=11, polyorder=2)
    else:
        effs_smooth = effs_sorted
    
    # Chart 1: True vs Predicted Efficiency Curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(flows_sorted, effs_smooth * 100, 'k-', linewidth=2.0, 
             label='True Efficiency', zorder=1)
    ax1.plot(true_bep['flow'], true_bep['efficiency'] * 100,
             marker='s', color=colors['True BEP'], markersize=10,
             markeredgecolor='black', markeredgewidth=1.5,
             label='True BEP', zorder=5)
    
    for opt_name, results in results_dict.items():
        flows = results['flows']
        effs = results['true_efficiencies'] * 100
        color = colors.get(opt_name, colors['ESC'])
        ax1.scatter(flows, effs, alpha=1, s=40, color=color, zorder=2)
        pred_bep = results['predicted_bep']
        ax1.plot(pred_bep['flow'], pred_bep['efficiency'] * 100,
                marker='v', color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'{opt_name} BEP', zorder=4)
    
    ax1.set_xlabel('Flow Rate (m³/h)', fontweight='bold')
    ax1.set_ylabel('Efficiency (%)', fontweight='bold')
    ax1.set_ylim(0, 80)
    # ax1.set_title(f"True vs Predicted Efficiency Curve (H = {target_head}m)", fontweight='bold', pad=15)
    ax1.legend(loc='upper right', frameon=True, shadow=False)
    # ax1.grid(False, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_curve_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'efficiency_curve_comparison.pdf'))
    plt.close()
    
    # Chart 2: Convergence Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for opt_name, results in results_dict.items():
        iterations = results['iterations']
        convergence = results['convergence']
        # Normalize convergence to efficiency scale
        convergence_normalized = (convergence - convergence.min()) / (convergence.max() - convergence.min())
        eff_min = min(results['true_efficiencies']) * 100
        eff_max = max(results['true_efficiencies']) * 100
        convergence_eff = eff_min + convergence_normalized * (eff_max - eff_min)
        color = colors.get(opt_name, colors['ESC'])
        ax2.plot(iterations, convergence_eff, linewidth=2.5, marker='o', markersize=4,
                 color=color, label=opt_name)
    ax2.axhline(y=true_bep['efficiency'] * 100, color=colors['True BEP'],
                linestyle='--', linewidth=2, label='True BEP Efficiency')
    ax2.set_xlabel('Evaluations (Iterations)', fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontweight='bold')
    # ax2.set_title(f"Convergence to BEP (H = {target_head}m)", fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, shadow=False)
    # ax2.grid(False, alpha=0.3)
    ax2.set_xlim(0, max(iterations))
    ax2.set_ylim(min(convergence_eff), 60)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_plot.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'convergence_plot.pdf'))
    plt.close()
    
    print("\n✓ Visualizations saved to:", output_dir)

def main():
    """
    Main function with user input for test parameters
    """
    print("="*80)
    print("STATIC HEAD BEP DETECTION TEST")
    print("Testing ESC with firstOrderProxy")
    print("="*80)
    
    # =========================================================================
    # USER INPUT SECTION
    # =========================================================================
    print("\n" + "="*80)
    print("TEST CONFIGURATION")
    print("="*80)
    
    # Get target head from user
    while True:
        try:
            target_head = float(input("\nEnter target head (m) [default: 30.0]: ") or "30.0")
            if 20.0 <= target_head <= 45.0:
                break
            else:
                print("⚠️  Warning: Head should be between 20-45m for this pump")
                confirm = input("Continue anyway? (y/n): ").lower()
                if confirm == 'y':
                    break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Get time per evaluation from user
    while True:
        try:
            time_per_eval = float(input("Enter time per iteration (minutes) [default: 2.5]: ") or "2.5")
            if time_per_eval > 0:
                break
            else:
                print("❌ Time must be positive.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Get max iterations from user
    while True:
        try:
            max_iterations = int(input("Enter max iterations [default: 30]: ") or "30")
            if max_iterations > 0:
                break
            else:
                print("❌ Iterations must be positive.")
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")
    
    # Get noise level from user
    while True:
        try:
            noise_level = float(input("Enter noise level (0.0-0.1) [default: 0.02]: ") or "0.02")
            if 0.0 <= noise_level <= 0.2:
                break
            else:
                print("⚠️  Warning: Noise level typically 0-0.1 (0-10%)")
                confirm = input("Continue anyway? (y/n): ").lower()
                if confirm == 'y':
                    break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY:")
    print("="*80)
    print(f"  Target Head:          {target_head} m")
    print(f"  Time per Iteration:   {time_per_eval} minutes")
    print(f"  Max Iterations:       {max_iterations}")
    print(f"  Noise Level:          {noise_level*100:.1f}%")
    print(f"  Estimated Total Time: {max_iterations * time_per_eval:.1f} minutes")
    print(f"  Frequency Range:      30-65 Hz")
    print("="*80)
    
    confirm = input("\nProceed with test? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Test cancelled.")
        return
    
    # =========================================================================
    # SETUP
    # =========================================================================
    output_dir = Path(f'results/static/esc/{int(target_head)}m')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pump and proxy
    pump = CommercialPumpSimulator(system_head=target_head, noise_level=noise_level)
    proxy = firstOrderProxy()
    
    # =========================================================================
    # FIND TRUE BEP
    # =========================================================================
    print("\n" + "="*80)
    print(f"FINDING TRUE BEP AT H = {target_head}m")
    print("="*80)
    true_bep = find_true_bep_at_constant_head(pump, target_head)
    
    print(f"\nTrue BEP Found:")
    print(f"  Flow:       {true_bep['flow']:.3f} m³/h")
    print(f"  Efficiency: {true_bep['efficiency']*100:.2f}%")
    print(f"  Frequency:  {true_bep['frequency']:.1f} Hz")
    
    # =========================================================================
    # RUN OPTIMIZER
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING ESC OPTIMIZER")
    print("="*80)
    optimizer = ExtremumSeekingControl(freq_min=10, freq_max=60, step_size=2.0, proxy_function=proxy)
    results = run_optimizer_test(optimizer, pump, target_head, max_iterations, time_per_eval)
    
    print(f"\nPredicted BEP:")
    print(f"  Flow:       {results['predicted_bep']['flow']:.3f} m³/h")
    print(f"  Efficiency: {results['predicted_bep']['efficiency']*100:.2f}%")
    print(f"  Frequency:  {results['predicted_bep']['frequency']:.1f} Hz")
    print(f"  Found at:   Iteration {results['predicted_bep']['iteration']}")
    print(f"  Time:       {results['predicted_bep']['time_minutes']:.1f} minutes")
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    metrics = calculate_metrics(true_bep, results['predicted_bep'], results, time_per_eval)
    
    print(f"\n" + "="*80)
    print("PERFORMANCE METRICS:")
    print("="*80)
    print(f"  Flow Error:           {metrics['relative_flow_error_pct']:.2f}%")
    print(f"  Efficiency Diff:      {metrics['efficiency_difference_pct']:.2f}%")
    print(f"  Evaluations to BEP:   {metrics['evaluations_to_bep']}")
    print(f"  Time to BEP:          {metrics['time_to_bep_minutes']:.1f} minutes")
    print(f"  Total Evaluations:    {metrics['total_evaluations']}")
    print(f"  Total Time:           {metrics['total_time_minutes']:.1f} minutes")
    print(f"  Converged:            {'Yes' if metrics['converged'] else 'No'}")
    
    # =========================================================================
    # CREATE VISUALIZATIONS
    # =========================================================================
    results_dict = {'ESC': results}
    create_visualizations(true_bep, results_dict, target_head,output_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'test_parameters': {
            'target_head': target_head,
            'max_iterations': max_iterations,
            'time_per_evaluation_minutes': time_per_eval,
            'noise_level': noise_level,
            'frequency_range': [30, 65],
            'proxy_function': 'firstOrderProxy'
        },
        'true_bep': {
            'flow': float(true_bep['flow']),
            'efficiency': float(true_bep['efficiency']),
            'frequency': float(true_bep['frequency'])
        },
        'optimizer': {
            'name': 'ESC',
            'predicted_bep': {
                'flow': float(results['predicted_bep']['flow']),
                'efficiency': float(results['predicted_bep']['efficiency']),
                'frequency': float(results['predicted_bep']['frequency']),
                'iteration': int(results['predicted_bep']['iteration']),
                'time_minutes': float(results['predicted_bep']['time_minutes'])
            },
            'metrics': {
                'relative_flow_error_pct': float(metrics['relative_flow_error_pct']),
                'efficiency_difference_pct': float(metrics['efficiency_difference_pct']),
                'evaluations_to_bep': int(metrics['evaluations_to_bep']),
                'time_to_bep_minutes': float(metrics['time_to_bep_minutes']),
                'total_evaluations': int(metrics['total_evaluations']),
                'total_time_minutes': float(metrics['total_time_minutes']),
                'converged': bool(metrics['converged'])
            }
        }
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'test_results.json'}")
    print("="*80)
    print(f"\n✅ Test completed successfully!")
    print(f"📁 All results saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()
