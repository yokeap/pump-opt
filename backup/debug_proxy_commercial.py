"""
Evaluation of Commercial Pump Model
Comparing Q²/P proxy with True Efficiency at different frequencies
All curves sweep to max flow, BEP calculated from each curve
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
from dataclasses import dataclass
from typing import Optional

# Import the pump model
from src.commercial_pump_model import CommercialPumpSimulator, PumpMeasurement
from src.proxy_functions import OriginalProxy

# Set Origin-style plotting parameters
def set_origin_style(ax, show_top=True, show_right=True):
    """Apply Origin software style to axes"""
    ax.spines['top'].set_visible(show_top)
    ax.spines['right'].set_visible(show_right)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    # Show ticks on all sides
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, labelsize=10,
                   top=show_top, right=show_right, bottom=True, left=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=3, width=1.0,
                   top=show_top, right=show_right, bottom=True, left=True)
    
    # Remove grid
    ax.grid(False)


def collect_pump_data(pump, freq_values, max_flow=5.0):
    """
    Collect pump data sweeping from 0 to max_flow for each frequency
    Calculate BEP from each curve
    
    Returns: dict with frequency as key, containing arrays and BEP info
    """
    data = {}
    bep_points = {}
    
    n_points = 50
    
    for freq in freq_values:
        flows = []
        powers = []
        true_effs = []
        q2p_values = []
        heads = []
        
        print(f"\nCollecting data for Frequency = {freq}Hz:")
        
        # Sweep through flow range
        flow_range = np.linspace(0.5, max_flow, n_points)
        
        for flow in flow_range:
            # Calculate head and efficiency directly from pump model
            head = pump._calculate_head(flow, freq)
            eff = pump._calculate_pump_efficiency(flow, freq)
            
            if head <= 0:
                continue
            
            # Calculate power from hydraulic equation
            rho = 1000  # kg/m³
            g = 9.81    # m/s²
            Q_m3s = flow / 3600  # Convert m³/h to m³/s
            P_hydraulic = (rho * g * Q_m3s * head) / 1000  # kW
            
            # Account for pump and motor efficiency
            if eff > 0.01:
                P_electrical = P_hydraulic / (eff * pump.motor_efficiency)
            else:
                P_electrical = 0.05
            
            P_electrical = max(0.01, P_electrical)
            
            # Calculate Q²/P proxy
            Q2P = (flow ** 2) / P_electrical
            
            flows.append(flow)
            powers.append(P_electrical)
            true_effs.append(eff * 100)  # Convert to %
            q2p_values.append(Q2P)
            heads.append(head)
        
        # Find BEP from true efficiency curve
        true_eff_bep_idx = np.argmax(true_effs)
        true_eff_bep_flow = flows[true_eff_bep_idx]
        true_eff_bep_value = true_effs[true_eff_bep_idx]
        
        # Find BEP from Q²/P curve
        q2p_bep_idx = np.argmax(q2p_values)
        q2p_bep_flow = flows[q2p_bep_idx]
        q2p_bep_value = q2p_values[q2p_bep_idx]
        
        data[freq] = {
            'flow': np.array(flows),
            'power': np.array(powers),
            'true_efficiency': np.array(true_effs),
            'Q2P': np.array(q2p_values),
            'head': np.array(heads)
        }
        
        bep_points[freq] = {
            'true_eff': {
                'flow': true_eff_bep_flow,
                'value': true_eff_bep_value
            },
            'q2p': {
                'flow': q2p_bep_flow,
                'value': q2p_bep_value
            }
        }
        
        print(f"  Collected {len(flows)} points from {min(flows):.2f} to {max(flows):.2f} m³/h")
        print(f"  True Eff BEP: Q={true_eff_bep_flow:.2f} m³/h, η={true_eff_bep_value:.2f}%")
        print(f"  Q²/P BEP: Q={q2p_bep_flow:.2f} m³/h, Q²/P={q2p_bep_value:.2f}")
    
    return data, bep_points


def plot_2d_true_efficiency(data, bep_points, freq_values):
    """2D plot: Flow vs True Efficiency with BEP marks"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, freq in enumerate(freq_values):
        d = data[freq]
        ax.plot(d['flow'], d['true_efficiency'], 
                color=colors[i], marker=markers[i], markersize=5,
                linewidth=2.5, label=f'{freq} Hz', markevery=5)
        
        # Mark BEP for true efficiency
        bep = bep_points[freq]['true_eff']
        ax.plot(bep['flow'], bep['value'], 
                marker='*', markersize=18, color=colors[i],
                markeredgecolor='black', markeredgewidth=1.5,
                linestyle='none', zorder=10)
    
    # Add BEP legend entry
    ax.plot([], [], marker='*', markersize=18, color='gray',
            markeredgecolor='black', markeredgewidth=1.5,
            linestyle='none', label='BEP')
    
    ax.set_xlabel('Flow (m³/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('True Pump Efficiency vs Flow', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, loc='best', fontsize=10)
    
    set_origin_style(ax)
    plt.tight_layout()
    
    return fig


def plot_2d_q2p_proxy(data, bep_points, freq_values):
    """2D plot: Flow vs Q²/P Proxy with BEP marks"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, freq in enumerate(freq_values):
        d = data[freq]
        ax.plot(d['flow'], d['Q2P'], 
                color=colors[i], marker=markers[i], markersize=5,
                linewidth=2.5, label=f'{freq} Hz', markevery=5)
        
        # Mark BEP for Q²/P
        bep = bep_points[freq]['q2p']
        ax.plot(bep['flow'], bep['value'], 
                marker='*', markersize=18, color=colors[i],
                markeredgecolor='black', markeredgewidth=1.5,
                linestyle='none', zorder=10)
    
    # Add BEP legend entry
    ax.plot([], [], marker='*', markersize=18, color='gray',
            markeredgecolor='black', markeredgewidth=1.5,
            linestyle='none', label='BEP')
    
    ax.set_xlabel('Flow (m³/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Q²/P Proxy (m⁶·h⁻²·kW⁻¹)', fontsize=12, fontweight='bold')
    ax.set_title('Q²/P Efficiency Proxy vs Flow', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, loc='best', fontsize=10)
    
    set_origin_style(ax)
    plt.tight_layout()
    
    return fig


def plot_3d_true_efficiency(data, bep_points, freq_values):
    """3D plot with curves only (no surface): Frequency vs Flow vs True Efficiency"""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot curves for each frequency
    for i, freq in enumerate(freq_values):
        d = data[freq]
        
        # Plot the curve
        ax.plot(np.full_like(d['flow'], freq), d['flow'], d['true_efficiency'],
                color=colors[i], linewidth=3, alpha=0.9, label=f'{freq} Hz')
        
        # Mark BEP on the curve
        bep = bep_points[freq]['true_eff']
        ax.scatter([freq], [bep['flow']], [bep['value']],
                  marker='*', s=400, c=colors[i], 
                  edgecolors='black', linewidths=2.5, zorder=100)
    
    # Adjust viewing angle for clarity (not 45 degrees)
    ax.view_init(elev=20, azim=125)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Flow (m³/h)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('True Efficiency (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('True Efficiency (★ = BEP)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    
    # Style the 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(1.5)
    ax.yaxis.pane.set_linewidth(1.5)
    ax.zaxis.pane.set_linewidth(1.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_3d_q2p_proxy(data, bep_points, freq_values):
    """3D plot with curves only (no surface): Frequency vs Flow vs Q²/P Proxy"""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot curves for each frequency
    for i, freq in enumerate(freq_values):
        d = data[freq]
        
        # Plot the curve
        ax.plot(np.full_like(d['flow'], freq), d['flow'], d['Q2P'],
                color=colors[i], linewidth=3, alpha=0.9, label=f'{freq} Hz')
        
        # Mark BEP on the curve
        bep = bep_points[freq]['q2p']
        ax.scatter([freq], [bep['flow']], [bep['value']],
                  marker='*', s=400, c=colors[i], 
                  edgecolors='black', linewidths=2.5, zorder=100)
    
    # Adjust viewing angle for clarity (not 45 degrees)
    ax.view_init(elev=20, azim=125)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Flow (m³/h)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Q²/P Proxy (m⁶·h⁻²·kW⁻¹)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Q²/P Efficiency Proxy (★ = BEP)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    
    # Style the 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(1.5)
    ax.yaxis.pane.set_linewidth(1.5)
    ax.zaxis.pane.set_linewidth(1.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def calculate_bep_comparison(bep_points, freq_values):
    """Compare BEP locations between true efficiency and Q²/P"""
    print("\n" + "="*70)
    print("BEP COMPARISON: True Efficiency vs Q²/P Proxy")
    print("="*70)
    print(f"{'Freq':<8} {'True η BEP':<15} {'Q²/P BEP':<15} {'Flow Error':<12}")
    print(f"{'(Hz)':<8} {'Flow (m³/h)':<15} {'Flow (m³/h)':<15} {'(m³/h)':<12}")
    print("-"*70)
    
    flow_errors = []
    
    for freq in freq_values:
        true_bep = bep_points[freq]['true_eff']['flow']
        q2p_bep = bep_points[freq]['q2p']['flow']
        error = abs(true_bep - q2p_bep)
        flow_errors.append(error)
        
        print(f"{freq:<8} {true_bep:<15.3f} {q2p_bep:<15.3f} {error:<12.3f}")
    
    print("-"*70)
    print(f"Mean Flow Error: {np.mean(flow_errors):.3f} m³/h")
    print(f"Max Flow Error:  {np.max(flow_errors):.3f} m³/h")
    print(f"Std Flow Error:  {np.std(flow_errors):.3f} m³/h")


def calculate_correlation(data, freq_values):
    """Calculate correlation between Q²/P and True Efficiency"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: Q²/P Proxy vs True Efficiency")
    print("="*70)
    
    all_true_eff = []
    all_q2p = []
    
    for freq in freq_values:
        d = data[freq]
        all_true_eff.extend(d['true_efficiency'])
        all_q2p.extend(d['Q2P'])
    
    all_true_eff = np.array(all_true_eff)
    all_q2p = np.array(all_q2p)
    
    # Normalize for correlation
    all_q2p_norm = (all_q2p - np.min(all_q2p)) / (np.max(all_q2p) - np.min(all_q2p))
    all_true_eff_norm = (all_true_eff - np.min(all_true_eff)) / (np.max(all_true_eff) - np.min(all_true_eff))
    
    # Calculate correlation coefficient
    corr = np.corrcoef(all_q2p_norm, all_true_eff_norm)[0, 1]
    
    print(f"\nPearson Correlation Coefficient: {corr:.4f}")
    print(f"R² Value: {corr**2:.4f}")
    
    if corr > 0.9:
        print("Interpretation: EXCELLENT correlation - Q²/P is a very good proxy")
    elif corr > 0.7:
        print("Interpretation: GOOD correlation - Q²/P is a reasonable proxy")
    elif corr > 0.5:
        print("Interpretation: MODERATE correlation - Q²/P has some utility")
    else:
        print("Interpretation: WEAK correlation - Q²/P may not be reliable")
    
    print("\nPer-Frequency Statistics:")
    for freq in freq_values:
        d = data[freq]
        q2p_norm = (d['Q2P'] - np.min(d['Q2P'])) / (np.max(d['Q2P']) - np.min(d['Q2P']))
        eff_norm = (d['true_efficiency'] - np.min(d['true_efficiency'])) / (np.max(d['true_efficiency']) - np.min(d['true_efficiency']))
        corr_f = np.corrcoef(q2p_norm, eff_norm)[0, 1]
        print(f"  {freq}Hz: R = {corr_f:.4f}, R² = {corr_f**2:.4f}")


def main():
    """Main evaluation function"""
    print("="*70)
    print("COMMERCIAL PUMP MODEL EVALUATION")
    print("Schneider SUB 15-0.5cv")
    print("Rated: 60Hz, H=35m, Q=3m³/h, η=54%")
    print("="*70)
    
    # Initialize pump
    pump = CommercialPumpSimulator(system_head=30.0, random_seed=42)
    
    # Define test conditions
    freq_values = [30, 40, 50, 60]
    max_flow = 5.0  # Sweep to 5 m³/h
    
    # Collect data
    print("\nCollecting pump data (sweep to max flow)...")
    data, bep_points = collect_pump_data(pump, freq_values, max_flow)
    
    # Compare BEP locations
    calculate_bep_comparison(bep_points, freq_values)
    
    # Calculate correlation
    calculate_correlation(data, freq_values)
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("  [1/4] 2D True Efficiency vs Flow")
    fig1 = plot_2d_true_efficiency(data, bep_points, freq_values)
    
    print("  [2/4] 2D Q²/P Proxy vs Flow")
    fig2 = plot_2d_q2p_proxy(data, bep_points, freq_values)
    
    print("  [3/4] 3D True Efficiency (Curves Only)")
    fig3 = plot_3d_true_efficiency(data, bep_points, freq_values)
    
    print("  [4/4] 3D Q²/P Proxy (Curves Only)")
    fig4 = plot_3d_q2p_proxy(data, bep_points, freq_values)
    
    print("\nAll plots generated successfully!")
    print("="*70)
    
    plt.show()
    
    return data, bep_points


if __name__ == "__main__":
    data, bep_points = main()