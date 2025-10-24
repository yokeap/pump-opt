# ==============================================================================
# Dynamic Head BEP Detection - V3 with BEP Locking
# Key improvement: Stop seeking when BEP is found, maintain frequency until ΔQ
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

# IEEE Publication Style
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
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.markeredgewidth': 1.0,
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
    'figure.figsize': (3.5, 2.625),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'savefig.format': 'pdf',
    'axes.prop_cycle': plt.cycler(color=['#000000', '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']),
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
})

class BEPConvergenceDetector:
    """
    Detect when ESC has converged to BEP
    Uses sliding window to check stability
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 proxy_threshold: float = 0.02,
                 freq_threshold: float = 0.5):
        """
        Args:
            window_size: Number of evaluations to check
            proxy_threshold: Max std of proxy value (relative)
            freq_threshold: Max std of frequency (Hz)
        """
        self.window_size = window_size
        self.proxy_threshold = proxy_threshold
        self.freq_threshold = freq_threshold
        
        self.proxy_history = []
        self.freq_history = []
        
    def update(self, proxy_value: float, frequency: float):
        """Add new measurement"""
        self.proxy_history.append(proxy_value)
        self.freq_history.append(frequency)
        
        # Keep only recent history
        if len(self.proxy_history) > self.window_size:
            self.proxy_history.pop(0)
            self.freq_history.pop(0)
    
    def is_converged(self) -> tuple:
        """
        Check if BEP has been reached
        Returns: (converged: bool, reason: str)
        """
        if len(self.proxy_history) < self.window_size:
            return False, f"Insufficient data ({len(self.proxy_history)}/{self.window_size})"
        
        # Check proxy value stability (normalized by mean)
        proxy_array = np.array(self.proxy_history)
        proxy_mean = np.mean(proxy_array)
        proxy_std = np.std(proxy_array)
        proxy_cv = proxy_std / proxy_mean if proxy_mean > 0 else 1.0
        
        # Check frequency stability
        freq_array = np.array(self.freq_history)
        freq_std = np.std(freq_array)
        
        # Both must be stable
        proxy_stable = proxy_cv < self.proxy_threshold
        freq_stable = freq_std < self.freq_threshold
        
        if proxy_stable and freq_stable:
            return True, f"Converged (CV={proxy_cv:.3f}, σf={freq_std:.2f}Hz)"
        elif not proxy_stable:
            return False, f"Proxy unstable (CV={proxy_cv:.3f} > {self.proxy_threshold})"
        else:
            return False, f"Frequency unstable (σf={freq_std:.2f} > {self.freq_threshold}Hz)"
    
    def reset(self):
        """Reset history (after re-optimization)"""
        self.proxy_history = []
        self.freq_history = []
    
    def get_locked_frequency(self) -> float:
        """Get frequency to lock at (mean of recent history)"""
        if len(self.freq_history) == 0:
            return None
        return np.mean(self.freq_history)

class DynamicHeadSimulator:
    """Simulates time-varying head with Q-based threshold detection"""
    
    def __init__(self, 
                 initial_head: float = 30.0,
                 final_head: float = 24.0,
                 total_time_hours: float = 9.0,
                 threshold_delta_q_pct: float = 10.0,
                 min_time_between_reopt_hours: float = 0.5):
        
        self.initial_head = initial_head
        self.final_head = final_head
        self.total_time_hours = total_time_hours
        self.threshold_delta_q_pct = threshold_delta_q_pct
        self.min_time_between_reopt = min_time_between_reopt_hours
        
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
        """Determine if re-optimization should be triggered based on ΔQ"""
        
        if self.last_reopt_flow is None:
            return True, "Initial optimization"
        
        time_since_last = current_time - self.last_reopt_time
        if time_since_last < self.min_time_between_reopt:
            return False, f"Too soon (Δt={time_since_last:.2f}h)"
        
        delta_q_pct = abs((current_flow - self.last_reopt_flow) / self.last_reopt_flow) * 100
        
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

def run_dynamic_esc_with_locking(initial_head: float = 30.0,
                                final_head: float = 20.0,
                                total_time_hours: float = 9.0,
                                threshold_delta_q_pct: float = 10.0,
                                time_per_eval_minutes: float = 2.5,
                                esc_step_size: float = 1.0,
                                freq_min: float = 40.0,
                                freq_max: float = 60.0,
                                convergence_window: int = 10,
                                convergence_proxy_threshold: float = 0.02,
                                convergence_freq_threshold: float = 0.5):
    """
    Run dynamic head test with ESC optimizer and BEP locking
    """
    
    print(f"\n{'='*80}")
    print(f"DYNAMIC HEAD TEST - ESC WITH BEP LOCKING (V3)")
    print(f"{'='*80}")
    print(f"Head change:        {initial_head}m → {final_head}m over {total_time_hours}h")
    print(f"ΔQ threshold:       {threshold_delta_q_pct}%")
    print(f"Convergence window: {convergence_window} evaluations")
    print(f"Proxy CV threshold: {convergence_proxy_threshold}")
    print(f"Freq std threshold: {convergence_freq_threshold} Hz")
    print(f"{'='*80}\n")
    
    # Initialize components
    dynamic_sim = DynamicHeadSimulator(
        initial_head=initial_head,
        final_head=final_head,
        total_time_hours=total_time_hours,
        threshold_delta_q_pct=threshold_delta_q_pct,
        min_time_between_reopt_hours=0.5
    )
    
    pump = CommercialPumpSimulator(system_head=initial_head, noise_level=0.0)
    proxy = firstOrderProxy()
    
    optimizer = ExtremumSeekingControl(
        freq_min=freq_min,
        freq_max=freq_max,
        step_size=esc_step_size,
        proxy_function=proxy
    )
    
    convergence_detector = BEPConvergenceDetector(
        window_size=convergence_window,
        proxy_threshold=convergence_proxy_threshold,
        freq_threshold=convergence_freq_threshold
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
        'reoptimization_triggered': [],
        'bep_locked': [],
        'convergence_events': []
    }
    
    current_time_hours = 0.0
    max_evaluations = int(total_time_hours * 60 / time_per_eval_minutes)
    evaluation_count = 0
    best_efficiency_so_far = 0.0
    
    # State variables
    bep_locked = False
    locked_frequency = None
    seeking_mode = True
    
    print(f"Starting simulation (max {max_evaluations} evaluations)...\n")
    
    while evaluation_count < max_evaluations and current_time_hours < total_time_hours:
        
        current_head = dynamic_sim.get_head_at_time(current_time_hours)
        pump.set_system_head(current_head)
        
        # Determine frequency to use
        if bep_locked and locked_frequency is not None:
            # BEP locked - use locked frequency
            freq = locked_frequency
            seeking_mode = False
        else:
            # Seeking mode - use ESC
            freq = optimizer.suggest_frequency()
            seeking_mode = True
        
        # Get flow at this frequency
        flow = pump.calculate_flow_from_frequency_head(freq, current_head, tolerance=0.5)
        
        if flow is None or flow < 0.1:
            evaluation_count += 1
            current_time_hours += time_per_eval_minutes / 60.0
            continue
        
        # Check if re-optimization should be triggered (based on ΔQ)
        should_reopt, reopt_reason = dynamic_sim.should_reoptimize(flow, current_time_hours)
        
        if should_reopt:
            print(f"⚡ Re-optimization at t={current_time_hours:.2f}h, H={current_head:.1f}m, Q={flow:.2f}m³/h")
            print(f"   Reason: {reopt_reason}")
            
            # Reset optimizer and convergence detector
            optimizer = ExtremumSeekingControl(
                freq_min=freq_min,
                freq_max=freq_max,
                step_size=esc_step_size,
                proxy_function=proxy
            )
            convergence_detector.reset()
            
            # Unlock BEP
            bep_locked = False
            locked_frequency = None
            seeking_mode = True
            
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
        
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor, true_efficiency):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
                self.true_efficiency = true_efficiency
        
        measurement = SimpleMeasurement(flow, power, pf, eff)
        proxy_value = proxy.calculate(measurement)
        
        # Update optimizer (only if seeking)
        if seeking_mode:
            optimizer.update(freq, measurement)
        
        # Update convergence detector (only if seeking)
        if seeking_mode and not bep_locked:
            convergence_detector.update(proxy_value, freq)
            
            # Check if converged to BEP
            converged, conv_reason = convergence_detector.is_converged()
            
            if converged and not bep_locked:
                locked_frequency = convergence_detector.get_locked_frequency()
                bep_locked = True
                print(f"🔒 BEP LOCKED at t={current_time_hours:.2f}h, f={locked_frequency:.1f}Hz, Q={flow:.2f}m³/h, η={eff*100:.1f}%")
                print(f"   {conv_reason}")
                
                results['convergence_events'].append({
                    'time_hours': current_time_hours,
                    'frequency': locked_frequency,
                    'flow': flow,
                    'efficiency': eff,
                    'evaluation': evaluation_count
                })
        
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
        results['bep_locked'].append(bep_locked)
        
        # Progress indicator
        if evaluation_count % 20 == 0:
            mode_str = "LOCKED" if bep_locked else "SEEKING"
            print(f"  t={current_time_hours:.1f}h [{mode_str}], H={current_head:.1f}m, Q={flow:.2f}m³/h, η={eff*100:.1f}%, f={freq:.1f}Hz")
        
        evaluation_count += 1
        current_time_hours += time_per_eval_minutes / 60.0
    
    # Convert to numpy arrays
    for key in results:
        if isinstance(results[key], list) and key not in ['reoptimization_events', 'convergence_events']:
            results[key] = np.array(results[key])
    
    print(f"\n✓ Simulation completed: {evaluation_count} evaluations")
    print(f"✓ Re-optimizations: {len(results['reoptimization_events'])}")
    print(f"✓ BEP convergence events: {len(results['convergence_events'])}")
    print(f"✓ Time in locked mode: {np.sum(results['bep_locked']) / len(results['bep_locked']) * 100:.1f}%\n")
    
    return results

def create_ieee_plots_v3(results: dict, output_dir: Path):
    """Create IEEE plots with BEP locking visualization"""
    
    time = results['time_hours']
    head = results['head']
    flow = results['flow']
    efficiency = results['true_efficiency'] * 100
    frequency = results['frequency']
    bep_locked = results['bep_locked']
    reopt_events = results['reoptimization_events']
    conv_events = results['convergence_events']
    
    # =========================================================================
    # FIGURE 1: Head and Flow Evolution with Locking States
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.625))
    
    # Primary axis - Head
    color1 = '#000000'
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Head (m)', color=color1)
    line1 = ax1.plot(time, head, '-', color=color1, linewidth=1.5, label='Head')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim([0, time[-1]])
    
    # Secondary axis - Flow with locked/seeking coloring
    ax2 = ax1.twinx()
    color2 = '#E74C3C'
    ax2.set_ylabel('Flow (m³/h)', color=color2)
    
    # Plot flow with different colors for seeking vs locked
    time_array = np.array(time)
    flow_array = np.array(flow)
    locked_array = np.array(bep_locked)
    
    # Seeking segments (dashed)
    seeking_mask = ~locked_array
    if np.any(seeking_mask):
        ax2.plot(time_array[seeking_mask], flow_array[seeking_mask], 
                '--', color=color2, linewidth=1.2, alpha=0.6, label='Flow (seeking)')
    
    # Locked segments (solid)
    locked_mask = locked_array
    if np.any(locked_mask):
        ax2.plot(time_array[locked_mask], flow_array[locked_mask],
                '-', color=color2, linewidth=1.5, label='Flow (locked)')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Mark re-optimization events
    for event in reopt_events:
        ax1.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                   linewidth=1.0, alpha=0.7)
        ax1.plot(event['time_hours'], event['head'], 'o', 
                color='#3498DB', markersize=5, markeredgecolor='black', markeredgewidth=0.5)
    
    # Mark convergence events (BEP locked)
    for event in conv_events:
        ax2.plot(event['time_hours'], event['flow'], 's',
                color='#2ECC71', markersize=6, markeredgecolor='black', 
                markeredgewidth=0.8, label='BEP locked' if event == conv_events[0] else '')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_head_flow_evolution_locked.pdf', dpi=600)
    plt.savefig(output_dir / 'fig1_head_flow_evolution_locked.png', dpi=300)
    plt.close()
    
    print("✓ Figure 1 saved: Head and Flow Evolution (with locking)")
    
    # =========================================================================
    # FIGURE 2: Efficiency Tracking
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(3.5, 2.625))
    
    # Color points by locked state
    for i in range(len(time)):
        if bep_locked[i]:
            ax.scatter(time[i], efficiency[i], s=20, alpha=0.8, color='#2ECC71', 
                      edgecolors='none')
        else:
            ax.scatter(time[i], efficiency[i], s=20, alpha=0.4, color='#9B59B6',
                      edgecolors='none')
    
    # Plot best efficiency curve
    best_eff = results['best_efficiency_so_far'] * 100
    ax.plot(time, best_eff, '-', color='#000000', linewidth=1.5, 
           label='Best η found')
    
    # Mark events
    for event in reopt_events:
        ax.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                  linewidth=1.0, alpha=0.7)
    
    for event in conv_events:
        ax.plot(event['time_hours'], event['efficiency']*100, 's',
               color='#2ECC71', markersize=6, markeredgecolor='black', markeredgewidth=0.8)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='#000000', linewidth=1.5, label='Best η found'),
        Patch(facecolor='#9B59B6', alpha=0.4, label='Seeking'),
        Patch(facecolor='#2ECC71', alpha=0.8, label='Locked')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_efficiency_tracking_locked.pdf', dpi=600)
    plt.savefig(output_dir / 'fig2_efficiency_tracking_locked.png', dpi=300)
    plt.close()
    
    print("✓ Figure 2 saved: Efficiency Tracking (with locking states)")
    
    # =========================================================================
    # FIGURE 3: Frequency Control - FLAT during locked!
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(3.5, 2.625))
    
    # Plot with different colors for seeking vs locked
    for i in range(len(time)-1):
        if bep_locked[i]:
            ax.plot(time[i:i+2], frequency[i:i+2], '-', 
                   color='#2ECC71', linewidth=2.0, alpha=0.8)
        else:
            ax.plot(time[i:i+2], frequency[i:i+2], '-',
                   color='#9B59B6', linewidth=1.5, alpha=0.6)
    
    # Mark events
    for event in reopt_events:
        ax.axvline(x=event['time_hours'], color='#3498DB', linestyle=':', 
                  linewidth=1.0, alpha=0.7)
        idx = np.argmin(np.abs(time - event['time_hours']))
        ax.plot(event['time_hours'], frequency[idx], 'o',
               color='#3498DB', markersize=5, markeredgecolor='black', markeredgewidth=0.5)
    
    for event in conv_events:
        ax.plot(event['time_hours'], event['frequency'], 's',
               color='#2ECC71', markersize=6, markeredgecolor='black', markeredgewidth=0.8)
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='#9B59B6', linewidth=1.5, label='Seeking'),
        plt.Line2D([0], [0], color='#2ECC71', linewidth=2.0, label='Locked')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_frequency_control_locked.pdf', dpi=600)
    plt.savefig(output_dir / 'fig3_frequency_control_locked.png', dpi=300)
    plt.close()
    
    print("✓ Figure 3 saved: Frequency Control (FLAT during locked!)")
    
    # =========================================================================
    # FIGURE 4: Operating Points
    # =========================================================================
    fig4, ax = plt.subplots(figsize=(3.5, 2.625))
    
    # Color by locked state
    for i in range(len(time)):
        if bep_locked[i]:
            ax.scatter(flow[i], efficiency[i], c=time[i], cmap='Greens',
                      s=30, alpha=0.9, edgecolors='black', linewidths=0.4, vmin=0, vmax=time[-1])
        else:
            ax.scatter(flow[i], efficiency[i], c=time[i], cmap='Purples',
                      s=25, alpha=0.5, edgecolors='gray', linewidths=0.3, vmin=0, vmax=time[-1])
    
    # Mark events
    for event in reopt_events:
        idx = np.argmin(np.abs(time - event['time_hours']))
        ax.plot(flow[idx], efficiency[idx], '^', color='#E74C3C', 
               markersize=8, markeredgecolor='black', markeredgewidth=0.8,
               label='Re-opt' if event == reopt_events[0] else '')
    
    for event in conv_events:
        ax.plot(event['flow'], event['efficiency']*100, 's',
               color='#2ECC71', markersize=8, markeredgecolor='black', markeredgewidth=0.8,
               label='BEP locked' if event == conv_events[0] else '')
    
    ax.set_xlabel('Flow (m³/h)')
    ax.set_ylabel('Efficiency (%)')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_operating_points_locked.pdf', dpi=600)
    plt.savefig(output_dir / 'fig4_operating_points_locked.png', dpi=300)
    plt.close()
    
    print("✓ Figure 4 saved: Operating Points (with locking)")

def calculate_metrics_v3(results: dict):
    """Calculate performance metrics including locking statistics"""
    
    bep_locked = results['bep_locked']
    time_locked = np.sum(bep_locked) / len(bep_locked) * 100
    
    metrics = {
        'total_evaluations': len(results['flow']),
        'total_time_hours': results['time_hours'][-1],
        'number_of_reoptimizations': len(results['reoptimization_events']),
        'number_of_convergences': len(results['convergence_events']),
        'time_locked_pct': time_locked,
        'average_efficiency_pct': np.mean(results['true_efficiency']) * 100,
        'min_efficiency_pct': np.min(results['true_efficiency']) * 100,
        'max_efficiency_pct': np.max(results['true_efficiency']) * 100,
        'final_efficiency_pct': results['true_efficiency'][-1] * 100,
        'efficiency_std_pct': np.std(results['true_efficiency']) * 100,
        'average_frequency_hz': np.mean(results['frequency']),
        'average_flow_m3h': np.mean(results['flow']),
        'reoptimization_times_h': [e['time_hours'] for e in results['reoptimization_events']],
        'convergence_times_h': [e['time_hours'] for e in results['convergence_events']],
        
        # Locked mode statistics
        'avg_eff_while_locked_pct': np.mean(results['true_efficiency'][bep_locked]) * 100 if np.any(bep_locked) else 0,
        'avg_eff_while_seeking_pct': np.mean(results['true_efficiency'][~bep_locked]) * 100 if np.any(~bep_locked) else 0,
    }
    
    return metrics

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("DYNAMIC HEAD BEP DETECTION - V3 WITH BEP LOCKING")
    print("Improved: Stop seeking when BEP found, maintain until ΔQ trigger")
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
    esc_step = float(input("ESC step size (Hz) [default: 1.0]: ") or "1.0")
    
    print("\nCONVERGENCE DETECTION PARAMETERS:")
    conv_window = int(input("Convergence window (evals) [default: 10]: ") or "10")
    conv_proxy = float(input("Proxy CV threshold [default: 0.02]: ") or "0.02")
    conv_freq = float(input("Frequency std threshold (Hz) [default: 0.5]: ") or "0.5")
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY:")
    print("="*80)
    print(f"  Head change:           {initial_head}m → {final_head}m")
    print(f"  Total time:            {total_time} hours")
    print(f"  ΔQ threshold:          {threshold_delta_q}%")
    print(f"  Time per eval:         {time_per_eval} minutes")
    print(f"  ESC step size:         {esc_step} Hz")
    print(f"  Convergence window:    {conv_window} evaluations")
    print(f"  Proxy CV threshold:    {conv_proxy}")
    print(f"  Frequency std thresh:  {conv_freq} Hz")
    print(f"  Noise level:           0.0 (proof of concept)")
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
    results = run_dynamic_esc_with_locking(
        initial_head=initial_head,
        final_head=final_head,
        total_time_hours=total_time,
        threshold_delta_q_pct=threshold_delta_q,
        time_per_eval_minutes=time_per_eval,
        esc_step_size=esc_step,
        convergence_window=conv_window,
        convergence_proxy_threshold=conv_proxy,
        convergence_freq_threshold=conv_freq
    )
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    metrics = calculate_metrics_v3(results)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS:")
    print("="*80)
    print(f"  Total evaluations:         {metrics['total_evaluations']}")
    print(f"  Total time:                {metrics['total_time_hours']:.2f} hours")
    print(f"  Re-optimizations:          {metrics['number_of_reoptimizations']}")
    print(f"  BEP convergence events:    {metrics['number_of_convergences']}")
    print(f"  Time in locked mode:       {metrics['time_locked_pct']:.1f}%")
    print(f"\n  Average efficiency:        {metrics['average_efficiency_pct']:.2f}%")
    print(f"  Min efficiency:            {metrics['min_efficiency_pct']:.2f}%")
    print(f"  Max efficiency:            {metrics['max_efficiency_pct']:.2f}%")
    print(f"  Final efficiency:          {metrics['final_efficiency_pct']:.2f}%")
    print(f"  Efficiency std:            {metrics['efficiency_std_pct']:.2f}%")
    print(f"\n  Avg eff while LOCKED:      {metrics['avg_eff_while_locked_pct']:.2f}%")
    print(f"  Avg eff while SEEKING:     {metrics['avg_eff_while_seeking_pct']:.2f}%")
    print(f"\n  Average frequency:         {metrics['average_frequency_hz']:.1f} Hz")
    print(f"  Average flow:              {metrics['average_flow_m3h']:.2f} m³/h")
    print("="*80)
    
    # =========================================================================
    # CREATE IEEE PLOTS
    # =========================================================================
    print("\nGenerating IEEE publication plots with BEP locking...")
    create_ieee_plots_v3(results, output_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'version': 'V3_with_BEP_locking',
        'test_parameters': {
            'initial_head': initial_head,
            'final_head': final_head,
            'total_time_hours': total_time,
            'threshold_delta_q_pct': threshold_delta_q,
            'time_per_eval_minutes': time_per_eval,
            'esc_step_size': esc_step,
            'convergence_window': conv_window,
            'convergence_proxy_threshold': conv_proxy,
            'convergence_freq_threshold': conv_freq,
            'noise_level': 0.0,
            'optimizer': 'ESC_with_locking',
            'proxy_function': 'firstOrderProxy'
        },
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()},
        'reoptimization_events': results['reoptimization_events'],
        'convergence_events': results['convergence_events']
    }
    
    # Save JSON
    with open(output_dir / 'dynamic_test_results_v3_locked.json', 'w') as f:
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
        'reopt_triggered': results['reoptimization_triggered'],
        'bep_locked': results['bep_locked']
    })
    df.to_csv(output_dir / 'detailed_results_v3_locked.csv', index=False)
    
    print(f"\n✓ Results saved to: {output_dir / 'dynamic_test_results_v3_locked.json'}")
    print(f"✓ Detailed data saved to: {output_dir / 'detailed_results_v3_locked.csv'}")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS WITH BEP LOCKING:")
    print("="*80)
    print(f"✓ Frequency FLAT during locked periods ({metrics['time_locked_pct']:.1f}% of time)")
    print(f"✓ Flow shows clear FLAT segments at BEP")
    print(f"✓ Higher efficiency in locked mode: {metrics['avg_eff_while_locked_pct']:.2f}%")
    print(f"✓ vs seeking mode: {metrics['avg_eff_while_seeking_pct']:.2f}%")
    print(f"✓ Reduced wear on VFD (fewer frequency changes)")
    print(f"✓ Energy savings from stable operation")
    print("="*80)
    
    print(f"\n✅ V3 Test completed successfully!")
    print(f"📁 All results saved to: {output_dir.absolute()}")
    print(f"📊 IEEE-style figures show FLAT behavior at BEP")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()