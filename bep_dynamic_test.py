# ==============================================================================
# Dynamic Head BEP Detection - Real-world Implementation
# Uses ΔQ threshold (only measurable parameter in practice)
# ESC optimizer only - proof of concept
# IEEE publication style plots
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions import firstOrderProxy
from src.optimizers import ExtremumSeekingControl

# ============================================================================
# IEEE PUBLICATION STYLE (Origin-like)
# ============================================================================
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'mathtext.fontset': 'dejavuserif',
    'mathtext.rm': 'serif',
    'axes.unicode_minus': False,
    
    # Line and marker styles
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.markeredgewidth': 1.0,
    
    # Axes and ticks
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    
    # Figure settings
    'figure.figsize': (3.5, 2.625),  # Single column IEEE (88mm × 66mm)
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'savefig.format': 'pdf',
    
    # Colors (IEEE style - grayscale friendly)
    'axes.prop_cycle': plt.cycler(color=['#000000', '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']),
    
    # Legend
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    
    # Grid
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
})

class DynamicHeadSimulator:
    """
    Simulates time-varying head with Q-based threshold detection
    (Real-world implementation - only Q is measurable)
    """
    
    def __init__(self, 
                 initial_head: float = 30.0,
                 final_head: float = 24.0,
                 total_time_hours: float = 9.0,
                 threshold_delta_q_pct: float = 10.0,
                 min_time_between_reopt_hours: float = 0.5):
        """
        Args:
            initial_head: Starting head (m)
            final_head: Final head (m)
            total_time_hours: Total simulation time (hours)
            threshold_delta_q_pct: Threshold as % change in Q (e.g., 10 = 10%)
            min_time_between_reopt_hours: Minimum time between re-optimizations
        """
        
        self.initial_head = initial_head
        self.final_head = final_head
        self.total_time_hours = total_time_hours
        self.threshold_delta_q_pct = threshold_delta_q_pct
        self.min_time_between_reopt = min_time_between_reopt_hours
        
        # State tracking
        self.last_reopt_time = 0.0
        self.last_reopt_flow = None
        self.optimization_events = []
        
    def get_head_at_time(self, time_hours: float) -> float:
        """Get head at given time (linear decrease)"""
        if time_hours >= self.total_time_hours:
            return self.final_head
        
        progress = time_hours / self.total_time_hours
        head = self.initial_head + progress * (self.final_head - self.initial_head)
        return head
    
    def should_reoptimize(self, current_flow: float, current_time: float) -> tuple:
        """
        Determine if re-optimization should be triggered based on ΔQ
        
        Returns:
            (should_reopt: bool, reason: str)
        """
        
        # First evaluation - always optimize
        if self.last_reopt_flow is None:
            return True, "Initial optimization"
        
        # Enforce minimum time between re-optimizations
        time_since_last = current_time - self.last_reopt_time
        if time_since_last < self.min_time_between_reopt:
            return False, f"Too soon (Δt={time_since_last:.2f}h < {self.min_time_between_reopt}h)"
        
        # Calculate relative change in flow
        delta_q_pct = abs((current_flow - self.last_reopt_flow) / self.last_reopt_flow) * 100
        
        # Check threshold
        if delta_q_pct >= self.threshold_delta_q_pct:
            return True, f"ΔQ={delta_q_pct:.1f}% ≥ {self.threshold_delta_q_pct}%"
        else:
            return False, f"ΔQ={delta_q_pct:.1f}% < {self.threshold_delta_q_pct}%"
    
    def register_optimization(self, flow: float, time_hours: float, head: float):
        """Register that an optimization occurred"""
        self.last_reopt_flow = flow
        self.last_reopt_time = time_hours
        self.optimization_events.append({
            'time_hours': time_hours,
            'flow': flow,
            'head': head
        })

def run_dynamic_esc_test(initial_head: float = 30.0,
                        final_head: float = 24.0,
                        total_time_hours: float = 9.0,
                        threshold_delta_q_pct: float = 10.0,
                        time_per_eval_minutes: float = 2.5,
                        esc_step_size: float = 2.0,
                        freq_min: float = 30.0,
                        freq_max: float = 65.0):
    """
    Run dynamic head test with ESC optimizer
    
    Returns:
        results: Dictionary with complete test history
    """
    
    print(f"\n{'='*80}")
    print(f"DYNAMIC HEAD TEST - ESC OPTIMIZER")
    print(f"{'='*80}")
    print(f"Head change:        {initial_head}m → {final_head}m over {total_time_hours}h")
    print(f"ΔQ threshold:       {threshold_delta_q_pct}%")
    print(f"Time per eval:      {time_per_eval_minutes} min")
    print(f"ESC step size:      {esc_step_size} Hz")
    print(f"{'='*80}\n")
    
    # Initialize simulator
    dynamic_sim = DynamicHeadSimulator(
        initial_head=initial_head,
        final_head=final_head,
        total_time_hours=total_time_hours,
        threshold_delta_q_pct=threshold_delta_q_pct,
        min_time_between_reopt_hours=0.5
    )
    
    # Initialize pump (no noise for proof of concept)
    pump = CommercialPumpSimulator(
        system_head=initial_head,
        noise_level=0.0
    )
    
    # Initialize ESC optimizer
    proxy = firstOrderProxy()
    optimizer = ExtremumSeekingControl(
        freq_min=freq_min,
        freq_max=freq_max,
        step_size=esc_step_size,
        proxy_function=proxy
    )
    
    # Results storage
    results = {
        'time_hours': [],
        'head': [],
        'frequency': [],
        'flow': [],
        'power': [],
        'power_factor': [],
        'proxy_value': [],
        'true_efficiency': [],
        'best_efficiency_so_far': [],
        'reoptimization_events': [],
        'reoptimization_triggered': []
    }
    
    current_time_hours = 0.0
    max_evaluations = int(total_time_hours * 60 / time_per_eval_minutes)
    evaluation_count = 0
    best_efficiency_so_far = 0.0
    
    print(f"Starting simulation (max {max_evaluations} evaluations)...\n")
    
    while evaluation_count < max_evaluations and current_time_hours < total_time_hours:
        
        # Get current head
        current_head = dynamic_sim.get_head_at_time(current_time_hours)
        
        # Update pump system head
        pump.set_system_head(current_head)
        
        # Suggest frequency from optimizer
        freq = optimizer.suggest_frequency()
        
        # Get flow at this frequency and head
        flow = pump.calculate_flow_from_frequency_head(freq, current_head, tolerance=0.5)
        
        if flow is None or flow < 0.1:
            evaluation_count += 1
            current_time_hours += time_per_eval_minutes / 60.0
            continue
        
        # Check if re-optimization should be triggered
        should_reopt, reason = dynamic_sim.should_reoptimize(flow, current_time_hours)
        
        if should_reopt:
            print(f"⚡ Re-optimization at t={current_time_hours:.2f}h, H={current_head:.1f}m, Q={flow:.2f}m³/h")
            print(f"   Reason: {reason}")
            
            # Reset optimizer
            optimizer = ExtremumSeekingControl(
                freq_min=freq_min,
                freq_max=freq_max,
                step_size=esc_step_size,
                proxy_function=proxy
            )
            
            # Register optimization
            dynamic_sim.register_optimization(flow, current_time_hours, current_head)
            
            results['reoptimization_events'].append({
                'time_hours': current_time_hours,
                'head': current_head,
                'flow': flow,
                'evaluation': evaluation_count
            })
        
        # Calculate performance
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
        
        # Update optimizer
        optimizer.update(freq, measurement)
        
        # Track best efficiency
        best_efficiency_so_far = max(best_efficiency_so_far, eff)
        
        # Store results
        results['time_hours'].append(current_time_hours)
        results['head'].append(current_head)
        results['frequency'].append(freq)
        results['flow'].append(flow)
        results['power'].append(power)
        results['power_factor'].append(pf)
        results['proxy_value'].append(proxy_value)
        results['true_efficiency'].append(eff)
        results['best_efficiency_so_far'].append(best_efficiency_so_far)
        results['reoptimization_triggered'].append(should_reopt)
        
        # Progress indicator
        if evaluation_count % 20 == 0:
            print(f"  t={current_time_hours:.1f}h, H={current_head:.1f}m, Q={flow:.2f}m³/h, η={eff*100:.1f}%")
        
        # Advance time
        evaluation_count += 1
        current_time_hours += time_per_eval_minutes / 60.0
    
    # Convert to numpy arrays
    for key in results:
        if isinstance(results[key], list) and key != 'reoptimization_events':
            results[key] = np.array(results[key])
    
    print(f"\n✓ Simulation completed: {evaluation_count} evaluations")
    print(f"✓ Re-optimizations triggered: {len(results['reoptimization_events'])}\n")
    
    return results

def create_ieee_plots(results: dict, output_dir: Path):
    """
    Create IEEE publication-quality plots (Origin style)
    """
    
    time = results['time_hours']
    head = results['head']
    flow = results['flow']
    efficiency = results['true_efficiency'] * 100
    frequency = results['frequency']
    reopt_events = results['reoptimization_events']
    
    # =========================================================================
    # FIGURE 1: Head and Flow Evolution (Double Y-axis)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.625))
    
    # Primary axis - Head
    color1 = '#000000'
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Head (m)', color=color1)
    line1 = ax1.plot(time, head, '-', color=color1, linewidth=1.5, label='Head')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim([0, time[-1]])
    
    # Secondary axis - Flow
    ax2 = ax1.twinx()
    color2 = '#E74C3C'
    ax2.set_ylabel('Flow (m³/h)', color=color2)
    line2 = ax2.plot(time, flow, '--', color=color2, linewidth=1.5, label='Flow')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Mark re-optimization events
    for event in reopt_events:
        ax1.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                   linewidth=1.0, alpha=0.7)
        ax1.plot(event['time_hours'], event['head'], 'o', 
                color='#3498DB', markersize=5, markeredgecolor='black', markeredgewidth=0.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_head_flow_evolution.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_head_flow_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 1 saved: Head and Flow Evolution")
    
    # =========================================================================
    # FIGURE 2: Efficiency Tracking
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(3.5, 2.625))
    
    # Plot efficiency at each evaluation
    ax.scatter(time, efficiency, s=20, alpha=0.6, color='#2ECC71', 
              edgecolors='none', label='Measured η')
    
    # Plot best efficiency curve
    best_eff = results['best_efficiency_so_far'] * 100
    ax.plot(time, best_eff, '-', color='#000000', linewidth=1.5, 
           label='Best η found')
    
    # Mark re-optimization events
    for event in reopt_events:
        ax.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                  linewidth=1.0, alpha=0.7)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xlim([0, time[-1]])
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_efficiency_tracking.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_efficiency_tracking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 2 saved: Efficiency Tracking")
    
    # =========================================================================
    # FIGURE 3: Frequency Control
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(3.5, 2.625))
    
    ax.plot(time, frequency, '-', color='#9B59B6', linewidth=1.5)
    
    # Mark re-optimization events
    for event in reopt_events:
        ax.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                  linewidth=1.0, alpha=0.7)
        ax.plot(event['time_hours'], 
               frequency[np.argmin(np.abs(time - event['time_hours']))],
               'o', color='#3498DB', markersize=5, 
               markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_frequency_control.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_frequency_control.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 3 saved: Frequency Control")
    
    # =========================================================================
    # FIGURE 4: Flow vs Efficiency (Operating Points)
    # =========================================================================
    fig4, ax = plt.subplots(figsize=(3.5, 2.625))
    
    # Color by time (grayscale for IEEE)
    scatter = ax.scatter(flow, efficiency, c=time, cmap='gray', 
                        s=25, alpha=0.8, edgecolors='black', linewidths=0.3)
    
    # Mark re-optimization events
    for event in reopt_events:
        idx = np.argmin(np.abs(time - event['time_hours']))
        ax.plot(flow[idx], efficiency[idx], '^', color='#E74C3C', 
               markersize=8, markeredgecolor='black', markeredgewidth=0.8,
               label='Re-opt' if event == reopt_events[0] else '')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (h)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel('Flow (m³/h)')
    ax.set_ylabel('Efficiency (%)')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_operating_points.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_operating_points.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 4 saved: Operating Points")

def calculate_metrics(results: dict):
    """Calculate performance metrics"""
    
    metrics = {
        'total_evaluations': len(results['flow']),
        'total_time_hours': results['time_hours'][-1],
        'number_of_reoptimizations': len(results['reoptimization_events']),
        'average_efficiency_pct': np.mean(results['true_efficiency']) * 100,
        'min_efficiency_pct': np.min(results['true_efficiency']) * 100,
        'max_efficiency_pct': np.max(results['true_efficiency']) * 100,
        'final_efficiency_pct': results['true_efficiency'][-1] * 100,
        'efficiency_std_pct': np.std(results['true_efficiency']) * 100,
        'average_frequency_hz': np.mean(results['frequency']),
        'average_flow_m3h': np.mean(results['flow']),
        'reoptimization_times_h': [e['time_hours'] for e in results['reoptimization_events']]
    }
    
    return metrics

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("DYNAMIC HEAD BEP DETECTION - PROOF OF CONCEPT")
    print("Real-world implementation using ΔQ threshold")
    print("="*80 + "\n")
    
    # =========================================================================
    # TEST CONFIGURATION
    # =========================================================================
    print("TEST PARAMETERS:")
    print("-" * 80)
    
    # Get user input
    initial_head = float(input("Initial head (m) [default: 30.0]: ") or "30.0")
    final_head = float(input("Final head (m) [default: 24.0]: ") or "24.0")
    total_time = float(input("Total time (hours) [default: 9.0]: ") or "9.0")
    threshold_delta_q = float(input("ΔQ threshold (%) [default: 10.0]: ") or "10.0")
    time_per_eval = float(input("Time per evaluation (min) [default: 2.5]: ") or "2.5")
    esc_step = float(input("ESC step size (Hz) [default: 2.0]: ") or "2.0")
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY:")
    print("="*80)
    print(f"  Head change:        {initial_head}m → {final_head}m")
    print(f"  Total time:         {total_time} hours")
    print(f"  ΔQ threshold:       {threshold_delta_q}%")
    print(f"  Time per eval:      {time_per_eval} minutes")
    print(f"  ESC step size:      {esc_step} Hz")
    print(f"  Noise level:        0.0 (proof of concept)")
    print(f"  Max evaluations:    ~{int(total_time * 60 / time_per_eval)}")
    print("="*80)
    
    confirm = input("\nProceed with test? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Test cancelled.")
        return
    
    # =========================================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================================
    output_dir = Path('results/dynamic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # RUN TEST
    # =========================================================================
    results = run_dynamic_esc_test(
        initial_head=initial_head,
        final_head=final_head,
        total_time_hours=total_time,
        threshold_delta_q_pct=threshold_delta_q,
        time_per_eval_minutes=time_per_eval,
        esc_step_size=esc_step
    )
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    metrics = calculate_metrics(results)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS:")
    print("="*80)
    print(f"  Total evaluations:      {metrics['total_evaluations']}")
    print(f"  Total time:             {metrics['total_time_hours']:.2f} hours")
    print(f"  Re-optimizations:       {metrics['number_of_reoptimizations']}")
    print(f"  Average efficiency:     {metrics['average_efficiency_pct']:.2f}%")
    print(f"  Min efficiency:         {metrics['min_efficiency_pct']:.2f}%")
    print(f"  Max efficiency:         {metrics['max_efficiency_pct']:.2f}%")
    print(f"  Final efficiency:       {metrics['final_efficiency_pct']:.2f}%")
    print(f"  Efficiency std:         {metrics['efficiency_std_pct']:.2f}%")
    print(f"  Average frequency:      {metrics['average_frequency_hz']:.1f} Hz")
    print(f"  Average flow:           {metrics['average_flow_m3h']:.2f} m³/h")
    print("="*80)
    
    # =========================================================================
    # CREATE IEEE PLOTS
    # =========================================================================
    print("\nGenerating IEEE publication plots...")
    create_ieee_plots(results, output_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'test_parameters': {
            'initial_head': initial_head,
            'final_head': final_head,
            'total_time_hours': total_time,
            'threshold_delta_q_pct': threshold_delta_q,
            'time_per_eval_minutes': time_per_eval,
            'esc_step_size': esc_step,
            'noise_level': 0.0,
            'optimizer': 'ESC',
            'proxy_function': 'firstOrderProxy'
        },
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()},
        'reoptimization_events': results['reoptimization_events']
    }
    
    # Save JSON
    with open(output_dir / 'dynamic_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed CSV
    import pandas as pd
    df = pd.DataFrame({
        'time_hours': results['time_hours'],
        'head_m': results['head'],
        'frequency_hz': results['frequency'],
        'flow_m3h': results['flow'],
        'power_kw': results['power'],
        'power_factor': results['power_factor'],
        'efficiency_pct': results['true_efficiency'] * 100,
        'proxy_value': results['proxy_value'],
        'reopt_triggered': results['reoptimization_triggered']
    })
    df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    print(f"\n✓ Results saved to: {output_dir / 'dynamic_test_results.json'}")
    print(f"✓ Detailed data saved to: {output_dir / 'detailed_results.csv'}")
    print("="*80)
    print(f"\n✅ Test completed successfully!")
    print(f"📁 All results saved to: {output_dir.absolute()}")
    print(f"📊 IEEE-style figures ready for publication")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
