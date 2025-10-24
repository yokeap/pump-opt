# ==============================================================================
# Flow Behavior Analysis - Understanding H-Q Relationship
# Simple test: How does flow change when head changes at constant frequency?
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.commercial_pump_model import CommercialPumpSimulator

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
    'axes.unicode_minus': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.figsize': (7, 4),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

def analyze_flow_vs_head_at_constant_frequency():
    """
    Analyze how flow changes when head changes at constant frequency
    This is the fundamental pump behavior we need to understand
    """
    
    print("\n" + "="*80)
    print("FLOW BEHAVIOR ANALYSIS AT CONSTANT FREQUENCY")
    print("="*80)
    
    # Parameters
    initial_head = 30.0  # m
    final_head = 20.0    # m
    total_time = 20.0    # hours
    
    # Test at multiple constant frequencies
    test_frequencies = [50, 55, 60]  # Hz
    
    # Time array
    n_points = 200
    time_hours = np.linspace(0, total_time, n_points)
    
    # Head changes linearly over time
    head_array = initial_head + (final_head - initial_head) * (time_hours / total_time)
    
    # Initialize pump
    pump = CommercialPumpSimulator(system_head=initial_head, noise_level=0.0)
    
    # Storage for results
    results = {}
    
    print(f"\nTesting flow behavior at constant frequencies:")
    print(f"  Head range: {initial_head}m → {final_head}m")
    print(f"  Time: {total_time} hours")
    print(f"  Frequencies: {test_frequencies} Hz\n")
    
    # For each frequency
    for freq in test_frequencies:
        print(f"Testing at f = {freq} Hz...")
        
        flows = []
        powers = []
        efficiencies = []
        
        for i, (t, H) in enumerate(zip(time_hours, head_array)):
            # Calculate flow at this head and frequency
            Q = pump.calculate_flow_from_frequency_head(freq, H, tolerance=0.5)
            
            if Q is None or Q < 0.1:
                flows.append(np.nan)
                powers.append(np.nan)
                efficiencies.append(np.nan)
                continue
            
            # Calculate efficiency and power
            eff = pump._calculate_pump_efficiency(Q, freq)
            power, _, pf = pump._calculate_electrical_power(Q, freq)
            
            flows.append(Q)
            powers.append(power)
            efficiencies.append(eff * 100)
            
            if i % 50 == 0:
                print(f"  t={t:.1f}h, H={H:.1f}m → Q={Q:.2f}m³/h, η={eff*100:.1f}%, P={power:.2f}kW")
        
        results[freq] = {
            'flow': np.array(flows),
            'power': np.array(powers),
            'efficiency': np.array(efficiencies)
        }
        print(f"  ✓ Completed f={freq}Hz\n")
    
    # =========================================================================
    # PLOT: Head and Flow vs Time
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    # Primary axis - Head
    color_head = '#000000'
    ax1.set_xlabel('Time (h)', fontweight='bold')
    ax1.set_ylabel('Head (m)', color=color_head, fontweight='bold')
    line_head = ax1.plot(time_hours, head_array, '-', color=color_head, 
                        linewidth=2.0, label='Head (decreasing)')
    ax1.tick_params(axis='y', labelcolor=color_head)
    ax1.set_xlim([0, total_time])
    ax1.set_ylim([final_head - 2, initial_head + 2])
    
    # Secondary axis - Flow
    ax2 = ax1.twinx()
    color_flows = ['#E74C3C', '#3498DB', '#2ECC71']
    ax2.set_ylabel('Flow (m³/h)', fontweight='bold')
    
    lines_flow = []
    for i, freq in enumerate(test_frequencies):
        flow = results[freq]['flow']
        line = ax2.plot(time_hours, flow, '-', color=color_flows[i], 
                       linewidth=1.8, label=f'Flow at {freq} Hz', alpha=0.8)
        lines_flow.append(line[0])
    
    ax2.tick_params(axis='y')
    
    # Combined legend
    all_lines = line_head + lines_flow
    labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, labels, loc='upper left', frameon=True, shadow=True)
    
    ax1.set_title('Flow Behavior vs Head Change at Constant Frequency', 
                  fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'flow_vs_head_constant_freq.pdf', dpi=600)
    plt.savefig(output_dir / 'flow_vs_head_constant_freq.png', dpi=300)
    plt.close()
    
    print(f"✓ Figure saved: flow_vs_head_constant_freq.pdf")
    
    # =========================================================================
    # PLOT: Efficiency vs Time (at constant frequencies)
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(7, 4))
    
    for i, freq in enumerate(test_frequencies):
        eff = results[freq]['efficiency']
        ax.plot(time_hours, eff, '-', color=color_flows[i], 
               linewidth=1.8, label=f'{freq} Hz', alpha=0.8)
    
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title('Efficiency vs Time at Constant Frequency (Head Decreasing)', 
                 fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_xlim([0, total_time])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_vs_time_constant_freq.pdf', dpi=600)
    plt.savefig(output_dir / 'efficiency_vs_time_constant_freq.png', dpi=300)
    plt.close()
    
    print(f"✓ Figure saved: efficiency_vs_time_constant_freq.pdf")
    
    # =========================================================================
    # PLOT: Power vs Time
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(7, 4))
    
    for i, freq in enumerate(test_frequencies):
        power = results[freq]['power']
        ax.plot(time_hours, power, '-', color=color_flows[i], 
               linewidth=1.8, label=f'{freq} Hz', alpha=0.8)
    
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Power (kW)', fontweight='bold')
    ax.set_title('Power Consumption vs Time at Constant Frequency', 
                 fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_xlim([0, total_time])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'power_vs_time_constant_freq.pdf', dpi=600)
    plt.savefig(output_dir / 'power_vs_time_constant_freq.png', dpi=300)
    plt.close()
    
    print(f"✓ Figure saved: power_vs_time_constant_freq.pdf")
    
    # =========================================================================
    # ANALYSIS SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY:")
    print("="*80)
    
    for freq in test_frequencies:
        flow = results[freq]['flow']
        eff = results[freq]['efficiency']
        power = results[freq]['power']
        
        # Remove NaN values
        valid_mask = ~np.isnan(flow)
        flow_valid = flow[valid_mask]
        eff_valid = eff[valid_mask]
        power_valid = power[valid_mask]
        
        if len(flow_valid) > 0:
            print(f"\nFrequency: {freq} Hz")
            print(f"  Flow:       {flow_valid[0]:.2f} → {flow_valid[-1]:.2f} m³/h "
                  f"(Δ = {flow_valid[-1] - flow_valid[0]:+.2f})")
            print(f"  Efficiency: {eff_valid[0]:.1f}% → {eff_valid[-1]:.1f}% "
                  f"(Δ = {eff_valid[-1] - eff_valid[0]:+.1f}%)")
            print(f"  Power:      {power_valid[0]:.2f} → {power_valid[-1]:.2f} kW "
                  f"(Δ = {power_valid[-1] - power_valid[0]:+.2f} kW)")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("1. When head DECREASES (30m → 20m):")
    print("   - Flow INCREASES (lower resistance)")
    print("   - Efficiency typically DECREASES (moving away from BEP)")
    print("   - Power may increase or decrease depending on operating point")
    print("\n2. At constant frequency:")
    print("   - Pump operates on a fixed pump curve")
    print("   - System curve shifts (lower head = lower resistance)")
    print("   - Operating point moves along pump curve")
    print("\n3. For optimization:")
    print("   - Need to ADJUST frequency when head changes")
    print("   - Goal: maintain operation near BEP")
    print("   - Proxy should detect when efficiency degrades")
    print("="*80)
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print("="*80 + "\n")
    
    return results, time_hours, head_array

if __name__ == "__main__":
    results, time_hours, head_array = analyze_flow_vs_head_at_constant_frequency()