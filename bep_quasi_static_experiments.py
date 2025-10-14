# ==============================================================================
# Quasi-Static Head BEP Detection Test
# Tests ESC optimizer tracking BEP under slowly changing head conditions
# Head profile: 35m → 30m (8h) → 25m (5h) → 30m (3h)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import json
from pathlib import Path
import sys
from scipy.interpolate import interp1d

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions import firstOrderProxy
from src.optimizers import ExtremumSeekingControl

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'axes.grid': False,
})

def create_head_profile(time_per_eval=2.5):
    """
    Create quasi-static head profile
    
    *** EDIT THIS FUNCTION TO CHANGE HEAD PROFILE ***
    
    Profile:
      - 35m → 30m over 8 hours
      - 30m → 25m over 5 hours  
      - 25m → 30m over 3 hours
      
    Args:
        time_per_eval: Time per evaluation in minutes
        
    Returns:
        time_hours: Array of time points in hours
        head_profile: Array of head values in meters
        phase_boundaries: Dict with phase information
    """
    
    # =========================================================================
    # EDIT THESE VALUES TO CHANGE HEAD PROFILE
    # =========================================================================
    
    # Phase 1: Start head → End head, Duration (hours)
    phase1_start = 35.0  # meters
    phase1_end = 30.0    # meters
    phase1_hours = 8.0   # hours
    
    # Phase 2: Start head → End head, Duration (hours)
    phase2_start = 30.0  # meters (should match phase1_end)
    phase2_end = 25.0    # meters
    phase2_hours = 5.0   # hours
    
    # Phase 3: Start head → End head, Duration (hours)
    phase3_start = 25.0  # meters (should match phase2_end)
    phase3_end = 30.0    # meters
    phase3_hours = 3.0   # hours
    
    # =========================================================================
    # END OF EDITABLE SECTION
    # =========================================================================
    
    # Convert to iterations
    iters_per_hour = 60 / time_per_eval
    
    # Phase durations in iterations
    phase1_iters = int(phase1_hours * iters_per_hour)
    phase2_iters = int(phase2_hours * iters_per_hour)
    phase3_iters = int(phase3_hours * iters_per_hour)
    
    total_iters = phase1_iters + phase2_iters + phase3_iters
    
    # Create time array in hours
    time_minutes = np.arange(0, total_iters) * time_per_eval
    time_hours = time_minutes / 60
    
    # Create head profile
    head_profile = np.zeros(total_iters)
    
    # Phase 1: linear ramp
    head_profile[:phase1_iters] = np.linspace(phase1_start, phase1_end, phase1_iters)
    
    # Phase 2: linear ramp
    start_idx = phase1_iters
    end_idx = phase1_iters + phase2_iters
    head_profile[start_idx:end_idx] = np.linspace(phase2_start, phase2_end, phase2_iters)
    
    # Phase 3: linear ramp
    start_idx = phase1_iters + phase2_iters
    head_profile[start_idx:] = np.linspace(phase3_start, phase3_end, phase3_iters)
    
    phase_boundaries = {
        'phase1_end': phase1_hours,
        'phase2_end': phase1_hours + phase2_hours,
        'phase3_end': phase1_hours + phase2_hours + phase3_hours,
        'total_hours': phase1_hours + phase2_hours + phase3_hours
    }
    
    return time_hours, head_profile, phase_boundaries

def find_true_bep_at_head(pump, head):
    """Find true BEP at a specific head"""
    frequencies = np.linspace(30, 65, 100)
    
    flows, efficiencies, valid_freqs = [], [], []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, head, tolerance=0.5)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        eff = pump._calculate_pump_efficiency(flow, freq)
        
        flows.append(flow)
        efficiencies.append(eff)
        valid_freqs.append(freq)
    
    if len(flows) == 0:
        return None
    
    flows = np.array(flows)
    efficiencies = np.array(efficiencies)
    valid_freqs = np.array(valid_freqs)
    
    bep_idx = np.argmax(efficiencies)
    
    return {
        'flow': flows[bep_idx],
        'efficiency': efficiencies[bep_idx],
        'frequency': valid_freqs[bep_idx]
    }

def run_quasi_static_test(time_per_eval=2.5, noise_level=0.02):
    """
    Run quasi-static test with ESC optimizer
    
    NOTE: ESC (Extremum Seeking Control) has inherent limitations for tracking:
    - It oscillates around the optimum by design (perturbation-based)
    - Trade-off between exploration (large steps) and exploitation (small steps)
    - Not ideal for fast-changing conditions
    
    For better tracking performance, consider:
    - Gradient-based methods (if gradient info available)
    - Bayesian optimization (TPE, GP)
    - Adaptive ESC with time-varying step size
    
    Args:
        time_per_eval: Time per evaluation in minutes
        noise_level: Measurement noise level (0-1)
        
    Returns:
        results: Dictionary with all test data
    """
    
    print("="*80)
    print("QUASI-STATIC HEAD BEP TRACKING TEST")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Time per evaluation: {time_per_eval} minutes")
    print(f"  Noise level:         {noise_level*100:.1f}%")
    print(f"  Head profile:")
    print(f"    Phase 1: 35m → 30m (8 hours)")
    print(f"    Phase 2: 30m → 25m (5 hours)")
    print(f"    Phase 3: 25m → 30m (3 hours)")
    print(f"  Total duration:      16 hours")
    print("="*80)
    
    # Create head profile
    time_hours, head_profile, phase_boundaries = create_head_profile(time_per_eval)
    
    print(f"\nTotal iterations: {len(time_hours)}")
    print(f"Estimated duration: {time_hours[-1]:.1f} hours")
    
    # Initialize pump and optimizer
    pump = CommercialPumpSimulator(system_head=35.0, noise_level=noise_level)
    proxy = firstOrderProxy()
    
    # Find initial BEP to start ESC near optimal point
    print("\nFinding initial BEP at 35m...")
    initial_bep = find_true_bep_at_head(pump, 35.0)
    initial_freq = initial_bep['frequency'] if initial_bep else 50.0
    print(f"Initial BEP: {initial_bep['flow']:.2f} m³/h at {initial_freq:.1f} Hz")
    
    # IMPROVED ESC SETTINGS for quasi-static tracking
    # Much smaller step size for smoother tracking
    optimizer = ExtremumSeekingControl(
        freq_min=30, 
        freq_max=65, 
        step_size=0.3,  # VERY SMALL steps for quasi-static tracking
        proxy_function=proxy
    )
    
    # Manually set initial frequency near BEP
    optimizer.current_frequency = initial_freq
    
    # Add moving average filter for proxy values
    proxy_history = []
    filter_window = 5  # Average last 5 measurements
    
    # Storage for results
    results = {
        'time_hours': [],
        'head_profile_actual': [],
        'frequencies': [],
        'flows': [],
        'powers': [],
        'efficiencies': [],
        'proxy_values': [],
        'true_bep_flows': [],
        'true_bep_efficiencies': [],
        'esc_errors_pct': [],
        'flow_errors_abs': []
    }
    
    # Store original profile for reference
    original_time_hours = time_hours.copy()
    original_head_profile = head_profile.copy()
    
    print("\nRunning quasi-static test...")
    
    # Run test
    for i, (time_h, head) in enumerate(zip(time_hours, head_profile)):
        
        # CRITICAL: Update pump system head for current quasi-static condition
        pump.set_system_head(head)
        pump.current_head = head  # Ensure internal state is updated
        
        # Get frequency from optimizer
        freq = optimizer.suggest_frequency()
        
        # Calculate flow at this frequency and CURRENT head
        # This should use the updated head from pump
        flow = pump.calculate_flow_from_frequency_head(freq, head, tolerance=0.5)
        
        if flow is None:
            # If no valid flow, try a different frequency
            freq = np.random.uniform(30, 65)
            flow = pump.calculate_flow_from_frequency_head(freq, head, tolerance=0.5)
            
            if flow is None:
                # Skip this iteration if still no valid flow
                continue
        
        # Get pump measurements
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        # Create measurement
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor, true_efficiency):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
                self.true_efficiency = true_efficiency
        
        measurement = SimpleMeasurement(flow, power, pf, eff)
        proxy_value = proxy.calculate(measurement)
        
        # Apply moving average filter to proxy value for smoother tracking
        proxy_history.append(proxy_value)
        if len(proxy_history) > filter_window:
            proxy_history.pop(0)
        proxy_filtered = np.mean(proxy_history)
        
        # Create filtered measurement for optimizer
        measurement_filtered = SimpleMeasurement(flow, power, pf, eff)
        
        # Update optimizer with filtered proxy (by updating the measurement)
        optimizer.update(freq, measurement_filtered)
        
        # Find true BEP at current head
        true_bep = find_true_bep_at_head(pump, head)
        
        if true_bep is not None:
            flow_error = abs(flow - true_bep['flow'])
            flow_error_pct = (flow_error / true_bep['flow']) * 100
            
            results['true_bep_flows'].append(true_bep['flow'])
            results['true_bep_efficiencies'].append(true_bep['efficiency'])
            results['flow_errors_abs'].append(flow_error)
            results['esc_errors_pct'].append(flow_error_pct)
        else:
            results['true_bep_flows'].append(np.nan)
            results['true_bep_efficiencies'].append(np.nan)
            results['flow_errors_abs'].append(np.nan)
            results['esc_errors_pct'].append(np.nan)
        
        # Store results (only for successful iterations)
        results['time_hours'].append(time_h)
        results['head_profile_actual'].append(head)
        results['frequencies'].append(freq)
        results['flows'].append(flow)
        results['powers'].append(power)
        results['efficiencies'].append(eff)
        results['proxy_values'].append(proxy_value)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(original_time_hours)} iterations "
                  f"({time_h:.1f}h, Head={head:.1f}m, Flow={flow:.2f}m³/h, "
                  f"Freq={freq:.1f}Hz, Error={results['esc_errors_pct'][-1]:.1f}%)")
    
    # Convert lists to arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    # Add original profile for visualization
    results['original_time_hours'] = original_time_hours
    results['original_head_profile'] = original_head_profile
    results['phase_boundaries'] = phase_boundaries
    
    print(f"\n✓ Test completed: {len(results['flows'])} successful iterations out of {len(original_time_hours)}")
    
    return results

def create_visualizations(results, output_dir):
    """Create all visualization charts"""
    
    time_hours = results['time_hours']
    head_profile = results['head_profile_actual']
    original_time_hours = results['original_time_hours']
    original_head_profile = results['original_head_profile']
    phase_boundaries = results['phase_boundaries']
    
    # =========================================================================
    # CHART 1: Head Profile (Raw vs Quasi-Static)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Raw step profile
    raw_times = [0, 8, 8, 13, 13, 16]
    raw_heads = [35, 35, 30, 30, 25, 25]
    ax1.plot(raw_times, raw_heads, 'r--', linewidth=2.5, alpha=0.6,
             label='Raw Step Changes', marker='s', markersize=8)
    
    # Quasi-static profile (original planned)
    ax1.plot(original_time_hours, original_head_profile, 'b-', linewidth=2.5,
             label='Quasi-Static Profile', alpha=0.7)
    
    # Actual executed profile
    ax1.plot(time_hours, head_profile, 'g.', markersize=2, alpha=0.5,
             label='Actual Executed Points')
    
    # Phase boundaries
    ax1.axvline(8, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(13, color='gray', linestyle=':', alpha=0.7)
    ax1.text(4, 36, 'Phase 1\n35→30m', ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(10.5, 36, 'Phase 2\n30→25m', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.text(14.5, 36, 'Phase 3\n25→30m', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax1.set_xlabel('Time (hours)', fontweight='bold')
    ax1.set_ylabel('Head (m)', fontweight='bold')
    ax1.set_title('Quasi-Static Head Profile', fontweight='bold', pad=15)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(23, 37)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart1_head_profile.png', dpi=300)
    plt.savefig(output_dir / 'chart1_head_profile.pdf')
    print(f"✓ Chart 1 saved: chart1_head_profile.png/.pdf")
    plt.close()
    
    # =========================================================================
    # CHART 2: BEP Tracking (Flow vs Time)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # True BEP
    ax2.plot(time_hours, results['true_bep_flows'], 'g-', linewidth=2.5,
             label='True BEP Flow', marker='o', markersize=3, markevery=20)
    
    # ESC tracked flow
    ax2.plot(time_hours, results['flows'], 'b-', linewidth=2, alpha=0.7,
             label='ESC Tracked Flow')
    
    # Phase boundaries
    ax2.axvline(8, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(13, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Time (hours)', fontweight='bold')
    ax2.set_ylabel('Flow (m³/h)', fontweight='bold')
    ax2.set_title('BEP Flow Tracking Performance', fontweight='bold', pad=15)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart2_bep_tracking.png', dpi=300)
    plt.savefig(output_dir / 'chart2_bep_tracking.pdf')
    print(f"✓ Chart 2 saved: chart2_bep_tracking.png/.pdf")
    plt.close()
    
    # =========================================================================
    # CHART 3: Tracking Error Over Time
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    ax3.plot(time_hours, results['esc_errors_pct'], 'r-', linewidth=2,
             label='Flow Error')
    ax3.axhline(5, color='orange', linestyle='--', alpha=0.7,
                label='5% Threshold')
    ax3.axhline(10, color='red', linestyle='--', alpha=0.7,
                label='10% Threshold')
    
    # Phase boundaries
    ax3.axvline(8, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(13, color='gray', linestyle=':', alpha=0.5)
    
    # Fill phases
    ax3.axvspan(0, 8, alpha=0.1, color='wheat')
    ax3.axvspan(8, 13, alpha=0.1, color='lightblue')
    ax3.axvspan(13, 16, alpha=0.1, color='lightgreen')
    
    ax3.set_xlabel('Time (hours)', fontweight='bold')
    ax3.set_ylabel('Flow Error (%)', fontweight='bold')
    ax3.set_title('ESC Tracking Error Over Time', fontweight='bold', pad=15)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 16)
    ax3.set_ylim(0, max(20, np.nanmax(results['esc_errors_pct']) * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart3_tracking_error.png', dpi=300)
    plt.savefig(output_dir / 'chart3_tracking_error.pdf')
    print(f"✓ Chart 3 saved: chart3_tracking_error.png/.pdf")
    plt.close()
    
    # =========================================================================
    # CHART 4: Efficiency Tracking
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    # True BEP efficiency
    ax4.plot(time_hours, results['true_bep_efficiencies'] * 100, 'g-', 
             linewidth=2.5, label='True BEP Efficiency', 
             marker='o', markersize=3, markevery=20)
    
    # ESC tracked efficiency
    ax4.plot(time_hours, results['efficiencies'] * 100, 'b-', 
             linewidth=2, alpha=0.7, label='ESC Achieved Efficiency')
    
    # Phase boundaries
    ax4.axvline(8, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(13, color='gray', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Time (hours)', fontweight='bold')
    ax4.set_ylabel('Efficiency (%)', fontweight='bold')
    ax4.set_title('Efficiency Tracking Performance', fontweight='bold', pad=15)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart4_efficiency_tracking.png', dpi=300)
    plt.savefig(output_dir / 'chart4_efficiency_tracking.pdf')
    print(f"✓ Chart 4 saved: chart4_efficiency_tracking.png/.pdf")
    plt.close()
    
    # =========================================================================
    # CHART 5: Phase-by-Phase Performance
    # =========================================================================
    fig5, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define phase indices
    phase_starts = [0, int(8 * 60 / (time_hours[1] * 60)), 
                   int(13 * 60 / (time_hours[1] * 60))]
    phase_ends = [int(8 * 60 / (time_hours[1] * 60)), 
                 int(13 * 60 / (time_hours[1] * 60)), 
                 len(time_hours)]
    phase_names = ['Phase 1: 35→30m', 'Phase 2: 30→25m', 'Phase 3: 25→30m']
    colors_phase = ['wheat', 'lightblue', 'lightgreen']
    
    for i, (ax, start, end, name, color) in enumerate(zip(axes, phase_starts, 
                                                          phase_ends, phase_names, 
                                                          colors_phase)):
        phase_time = time_hours[start:end]
        phase_true = results['true_bep_flows'][start:end]
        phase_esc = results['flows'][start:end]
        
        ax.plot(phase_time, phase_true, 'g-', linewidth=2.5, 
               label='True BEP', marker='o', markersize=4, markevery=10)
        ax.plot(phase_time, phase_esc, 'b-', linewidth=2, alpha=0.7,
               label='ESC Tracked')
        
        ax.set_facecolor(color)
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel('Flow (m³/h)', fontweight='bold')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chart5_phase_performance.png', dpi=300)
    plt.savefig(output_dir / 'chart5_phase_performance.pdf')
    print(f"✓ Chart 5 saved: chart5_phase_performance.png/.pdf")
    plt.close()

def calculate_statistics(results):
    """Calculate performance statistics"""
    
    # Overall statistics
    mean_error_pct = np.nanmean(results['esc_errors_pct'])
    max_error_pct = np.nanmax(results['esc_errors_pct'])
    std_error_pct = np.nanstd(results['esc_errors_pct'])
    
    # Calculate phase statistics
    time_hours = results['time_hours']
    errors = results['esc_errors_pct']
    
    phase_starts = [0, int(8 * len(time_hours) / 16), int(13 * len(time_hours) / 16)]
    phase_ends = [int(8 * len(time_hours) / 16), int(13 * len(time_hours) / 16), len(time_hours)]
    
    phase_stats = []
    for i, (start, end) in enumerate(zip(phase_starts, phase_ends)):
        phase_errors = errors[start:end]
        phase_stats.append({
            'phase': i + 1,
            'mean_error_pct': float(np.nanmean(phase_errors)),
            'max_error_pct': float(np.nanmax(phase_errors)),
            'std_error_pct': float(np.nanstd(phase_errors))
        })
    
    return {
        'overall': {
            'mean_error_pct': float(mean_error_pct),
            'max_error_pct': float(max_error_pct),
            'std_error_pct': float(std_error_pct),
            'total_iterations': len(results['flows']),
            'duration_hours': float(time_hours[-1])
        },
        'phases': phase_stats
    }

def main():
    """Main function"""
    
    # Configuration
    time_per_eval = 2.5  # minutes
    noise_level = 0.0    # No noise for quasi-static test
    
    # Create output directory
    output_dir = Path('results/quasi_static')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run test
    results = run_quasi_static_test(time_per_eval, noise_level)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(results, output_dir)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("CALCULATING STATISTICS")
    print("="*80)
    stats = calculate_statistics(results)
    
    print(f"\nOverall Performance:")
    print(f"  Mean Error:  {stats['overall']['mean_error_pct']:.2f}%")
    print(f"  Max Error:   {stats['overall']['max_error_pct']:.2f}%")
    print(f"  Std Error:   {stats['overall']['std_error_pct']:.2f}%")
    print(f"  Duration:    {stats['overall']['duration_hours']:.1f} hours")
    
    print(f"\nPhase-by-Phase Performance:")
    for phase in stats['phases']:
        print(f"  Phase {phase['phase']}: Mean={phase['mean_error_pct']:.2f}%, "
              f"Max={phase['max_error_pct']:.2f}%, Std={phase['std_error_pct']:.2f}%")
    
    # Save results
    save_data = {
        'test_parameters': {
            'time_per_eval_minutes': time_per_eval,
            'noise_level': noise_level,
            'head_profile': '35m→30m(8h)→25m(5h)→30m(3h)'
        },
        'statistics': stats
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'test_results.json'}")
    print(f"✓ All files saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()