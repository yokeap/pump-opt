# ==============================================================================
# FILE: complete_analysis_charts.py
# Complete Publication-Quality BEP Detection Analysis
# Charts 1-4 with all data saved to results/debug_commercial/
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions import firstOrderProxy
import json
import pandas as pd
from pathlib import Path

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
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
    'axes.prop_cycle': plt.cycler(color=['#0072BD', '#D95319', '#77AC30']),
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'axes.grid': False,
})

def collect_data_at_constant_head(pump, proxy, target_head, freq_range=(30, 65, 100)):
    """Collect data at a single constant head (for Charts 1 & 2)"""
    frequencies = np.linspace(*freq_range)
    
    flows, true_effs, proxy_values, valid_freqs = [], [], [], []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.1)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
        
        measurement = SimpleMeasurement(flow, power, pf)
        proxy_value = proxy.calculate(measurement)
        
        flows.append(flow)
        true_effs.append(eff)
        proxy_values.append(proxy_value)
        valid_freqs.append(freq)
    
    return np.array(flows), np.array(true_effs), np.array(proxy_values), np.array(valid_freqs)

def collect_frequency_flow_data(pump, proxy, target_head, freq_range=(30, 65, 100)):
    """Collect frequency vs flow data at constant head (for Chart 3)"""
    frequencies = np.linspace(*freq_range)
    
    flows, true_effs, proxy_values, valid_freqs = [], [], [], []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.1)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
        
        measurement = SimpleMeasurement(flow, power, pf)
        proxy_value = proxy.calculate(measurement)
        
        flows.append(flow)
        true_effs.append(eff)
        proxy_values.append(proxy_value)
        valid_freqs.append(freq)
    
    return np.array(valid_freqs), np.array(flows), np.array(true_effs), np.array(proxy_values)

def create_complete_analysis():
    """
    Create all 4 publication-quality charts:
    1. True Efficiency vs Flow at constant heads
    2. Q/P Proxy vs Flow at constant heads
    3. Frequency vs Flow at constant heads
    4. BEP Correlation: True vs Proxy
    
    All results saved to: results/debug_commercial/
    """
    
    # Create output directory
    output_dir = Path("results/debug_commercial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPLETE BEP DETECTION ANALYSIS - ALL CHARTS")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Initialize
    pump = CommercialPumpSimulator(system_head=30.0, noise_level=0.00)
    proxy = firstOrderProxy()

    # Test heads
    target_heads = [20, 25, 30]
    colors = ['#0072BD', '#D95319', '#77AC30', '#EDB120']
    markers = ['o', 's', '^', 'd']
    
    # Collect all data for Charts 1 & 2
    all_data_ch12 = {}
    chart12_raw_data = []
    
    for head in target_heads:
        flows, effs, proxies, freqs = collect_data_at_constant_head(pump, proxy, head)
        
        # Find BEP for each head
        bep_idx = np.argmax(effs)
        proxy_bep_idx = np.argmax(proxies)
        
        all_data_ch12[head] = {
            'flows': flows,
            'effs': effs,
            'proxies': proxies,
            'freqs': freqs,
            'bep_flow': flows[bep_idx],
            'bep_eff': effs[bep_idx],
            'proxy_bep_flow': flows[proxy_bep_idx],
            'proxy_bep_value': proxies[proxy_bep_idx]
        }
        
        # Store for CSV
        for j in range(len(flows)):
            chart12_raw_data.append({
                'head_m': head,
                'flow_m3h': flows[j],
                'frequency_hz': freqs[j],
                'true_efficiency': effs[j],
                'proxy_value': proxies[j]
            })
        
        error = abs(flows[proxy_bep_idx] - flows[bep_idx])
        error_pct = (error / flows[bep_idx]) * 100
        
        print(f"\nHead = {head}m:")
        print(f"  True BEP:  {flows[bep_idx]:.3f} m³/h (η={effs[bep_idx]*100:.1f}%)")
        print(f"  Proxy BEP: {flows[proxy_bep_idx]:.3f} m³/h")
        print(f"  Error:     {error:.3f} m³/h ({error_pct:.1f}%)")
    
    # Save Chart 1&2 raw data
    df_ch12 = pd.DataFrame(chart12_raw_data)
    csv_ch12 = output_dir / "chart12_efficiency_curves_data.csv"
    df_ch12.to_csv(csv_ch12, index=False)
    print(f"\n✓ Chart 1&2 data saved: {csv_ch12}")
    
    # =========================================================================
    # CHART 1: TRUE EFFICIENCY
    # =========================================================================
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    
    for i, head in enumerate(target_heads):
        data = all_data_ch12[head]
        
        # Plot efficiency curve
        ax1.plot(data['flows'], data['effs']*100, 
                color=colors[i], linewidth=1.5, 
                label=f'{head} m', zorder=2)
        
        # Mark BEP with square
        ax1.plot(data['bep_flow'], data['bep_eff']*100, 
                marker='s', color=colors[i], markersize=7, 
                markeredgewidth=1.5, zorder=3)
    
    ax1.set_xlabel('Flow (m³/h)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
    # ax1.set_title('True Pump Efficiency at Constant Head', 
    #              fontsize=14, fontweight='bold', pad=15)
    ax1.legend(title='Head', loc='upper right', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 60)
    ax1.grid(False)
    
    for spine in ax1.spines.values():
        spine.set_visible(True)
    ax1.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    fig1_png = output_dir / "chart1_true_efficiency.png"
    fig1_pdf = output_dir / "chart1_true_efficiency.pdf"
    plt.savefig(fig1_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig1_pdf, bbox_inches='tight')
    print(f"\n✓ Chart 1 saved: {fig1_png}")
    print(f"✓ Chart 1 saved: {fig1_pdf}")
    
    # =========================================================================
    # CHART 2: Q/P PROXY
    # =========================================================================
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    for i, head in enumerate(target_heads):
        data = all_data_ch12[head]
        
        # Plot proxy curve
        ax2.plot(data['flows'], data['proxies'], 
                color=colors[i], linewidth=2.5, 
                label=f'{head} m', zorder=2)
        
        # Mark proxy BEP with cross
        ax2.plot(data['proxy_bep_flow'], data['proxy_bep_value'], 
                marker='+', color=colors[i], markersize=14, 
                markeredgewidth=3, zorder=3)
    
    ax2.set_xlabel('Flow (m³/h)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Q/P Proxy', fontsize=13, fontweight='bold')
    # ax2.set_title('Linear Q/P Proxy at Constant Head', 
    #              fontsize=14, fontweight='bold', pad=15)
    ax2.legend(title='Head', loc='upper right', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, np.max([all_data_ch12[h]['proxies'].max() for h in target_heads])*1.1)
    ax2.grid(False)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
    ax2.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    fig2_png = output_dir / "chart2_qp_proxy.png"
    fig2_pdf = output_dir / "chart2_qp_proxy.pdf"
    plt.savefig(fig2_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig2_pdf, bbox_inches='tight')
    print(f"✓ Chart 2 saved: {fig2_png}")
    print(f"✓ Chart 2 saved: {fig2_pdf}")
    
    # =========================================================================
    # CHART 3: FREQUENCY vs FLOW
    # =========================================================================
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    
    all_freq_data = {}
    chart3_raw_data = []
    
    for i, head in enumerate(target_heads):
        freqs, flows, effs, proxies = collect_frequency_flow_data(pump, proxy, head)
        all_freq_data[head] = {'freqs': freqs, 'flows': flows, 'effs': effs, 'proxies': proxies}
        
        # Store raw data for export
        for j in range(len(freqs)):
            chart3_raw_data.append({
                'head_m': head,
                'frequency_hz': freqs[j],
                'flow_m3h': flows[j],
                'true_efficiency': effs[j],
                'proxy_value': proxies[j]
            })
        
        # Plot frequency vs flow
        ax3.plot(freqs, flows, color=colors[i], linewidth=2.5, 
                marker=markers[i], markersize=5, markevery=5,
                label=f'{head} m', zorder=2)
        
        print(f"\nChart 3 - Head = {head}m:")
        print(f"  Flow range: {flows.min():.2f} - {flows.max():.2f} m³/h")
        print(f"  Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    
    # Save Chart 3 raw data
    df_ch3 = pd.DataFrame(chart3_raw_data)
    csv_ch3 = output_dir / "chart3_frequency_flow_data.csv"
    df_ch3.to_csv(csv_ch3, index=False)
    print(f"\n✓ Chart 3 data saved: {csv_ch3}")
    
    ax3.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Flow (m³/h)', fontsize=13, fontweight='bold')
    # ax3.set_title('Flow vs Frequency at Constant Head', 
    #              fontsize=14, fontweight='bold', pad=15)
    ax3.legend(title='Head', loc='upper left', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax3.set_xlim(30, 70)
    ax3.set_ylim(0, 5)
    ax3.grid(False)
    
    for spine in ax3.spines.values():
        spine.set_visible(True)
    ax3.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    fig3_png = output_dir / "chart3_frequency_vs_flow.png"
    fig3_pdf = output_dir / "chart3_frequency_vs_flow.pdf"
    plt.savefig(fig3_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig3_pdf, bbox_inches='tight')
    print(f"\n✓ Chart 3 saved: {fig3_png}")
    print(f"✓ Chart 3 saved: {fig3_pdf}")
    
    # =========================================================================
    # CHART 4: BEP CORRELATION PLOT
    # =========================================================================
    fig4 = plt.figure(figsize=(8, 8))
    ax4 = fig4.add_subplot(111)
    
    bep_true_flows = []
    bep_proxy_flows = []
    bep_data = []
    
    print(f"\n{'='*80}")
    print("BEP DETECTION COMPARISON:")
    print(f"{'='*80}")
    
    for i, head in enumerate(target_heads):
        data = all_freq_data[head]
        
        # Find BEP for true efficiency
        bep_true_idx = np.argmax(data['effs'])
        bep_true_flow = data['flows'][bep_true_idx]
        bep_true_eff = data['effs'][bep_true_idx]
        bep_true_freq = data['freqs'][bep_true_idx]
        
        # Find BEP for proxy
        bep_proxy_idx = np.argmax(data['proxies'])
        bep_proxy_flow = data['flows'][bep_proxy_idx]
        bep_proxy_value = data['proxies'][bep_proxy_idx]
        bep_proxy_freq = data['freqs'][bep_proxy_idx]
        
        bep_true_flows.append(bep_true_flow)
        bep_proxy_flows.append(bep_proxy_flow)
        
        error = abs(bep_proxy_flow - bep_true_flow)
        error_pct = (error / bep_true_flow) * 100
        
        bep_data.append({
            'head_m': head,
            'true_bep_flow_m3h': bep_true_flow,
            'true_bep_efficiency': bep_true_eff,
            'true_bep_frequency_hz': bep_true_freq,
            'proxy_bep_flow_m3h': bep_proxy_flow,
            'proxy_bep_value': bep_proxy_value,
            'proxy_bep_frequency_hz': bep_proxy_freq,
            'absolute_error_m3h': error,
            'percentage_error': error_pct
        })
        
        print(f"\nHead = {head}m:")
        print(f"  True BEP:  Q = {bep_true_flow:.3f} m³/h (η = {bep_true_eff*100:.1f}%) @ {bep_true_freq:.1f} Hz")
        print(f"  Proxy BEP: Q = {bep_proxy_flow:.3f} m³/h @ {bep_proxy_freq:.1f} Hz")
        print(f"  Error:     {error:.3f} m³/h ({error_pct:.1f}%)")
        
        # Plot BEP point
        ax4.scatter(bep_true_flow, bep_proxy_flow, 
                   color=colors[i], marker=markers[i], s=200, 
                   alpha=0.7, edgecolors='black', linewidth=2,
                   label=f'{head} m', zorder=2)
    
    # Save BEP data to CSV
    df_bep = pd.DataFrame(bep_data)
    bep_csv_path = output_dir / "chart4_bep_comparison_data.csv"
    df_bep.to_csv(bep_csv_path, index=False)
    print(f"\n✓ Chart 4 data saved: {bep_csv_path}")
    
    bep_true_flows = np.array(bep_true_flows)
    bep_proxy_flows = np.array(bep_proxy_flows)
    
    # Calculate correlation metrics
    pearson_r, pearson_p = pearsonr(bep_true_flows, bep_proxy_flows)
    r_squared = r2_score(bep_true_flows, bep_proxy_flows)
    
    # Fit linear regression line
    coeffs = np.polyfit(bep_true_flows, bep_proxy_flows, 1)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(bep_true_flows.min()*0.95, bep_true_flows.max()*1.05, 100)
    y_fit = poly(x_fit)
    
    ax4.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.8, 
            label='Linear fit', zorder=1)
    ax4.plot(x_fit, x_fit, 'r:', linewidth=2, alpha=0.6, 
            label='Perfect correlation', zorder=0)
    
    textstr = f'Pearson r = {pearson_r:.4f}\nR² = {r_squared:.4f}'
    if pearson_p < 0.001:
        textstr += '\np < 0.001'
    else:
        textstr += f'\np = {pearson_p:.3f}'
    
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax4.set_xlabel('True BEP Flow (m³/h)', fontsize=13, fontweight='normal')
    ax4.set_ylabel('Proxy BEP Flow (m³/h)', fontsize=13, fontweight='normal')
    # ax4.set_title('BEP Detection: True Efficiency vs Q/P Proxy', 
    #              fontsize=14, fontweight='bold', pad=15)
    ax4.legend(loc='lower right', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax4.grid(False)
    
    for spine in ax4.spines.values():
        spine.set_visible(True)
    ax4.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, top=True, right=True)
    
    all_flows = np.concatenate([bep_true_flows, bep_proxy_flows])
    flow_min = all_flows.min() * 0.95
    flow_max = all_flows.max() * 1.05
    ax4.set_xlim(flow_min, flow_max)
    ax4.set_ylim(flow_min, flow_max)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    fig4_png = output_dir / "chart4_bep_correlation.png"
    fig4_pdf = output_dir / "chart4_bep_correlation.pdf"
    plt.savefig(fig4_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig4_pdf, bbox_inches='tight')
    print(f"\n✓ Chart 4 saved: {fig4_png}")
    print(f"✓ Chart 4 saved: {fig4_pdf}")
    
    plt.show()
    
    # =========================================================================
    # STATISTICAL SUMMARY
    # =========================================================================
    mae = np.mean(np.abs(bep_proxy_flows - bep_true_flows))
    mape = np.mean(np.abs((bep_proxy_flows - bep_true_flows) / bep_true_flows)) * 100
    rmse = np.sqrt(np.mean((bep_proxy_flows - bep_true_flows)**2))
    
    print(f"\n{'='*80}")
    print("BEP CORRELATION ANALYSIS:")
    print(f"{'='*80}")
    print(f"\nBEP Flow Comparison (4 head conditions):")
    print(f"  Pearson correlation (r):  {pearson_r:.4f}")
    print(f"  R-squared (R²):           {r_squared:.4f}")
    print(f"  p-value:                  {pearson_p:.4f}")
    print(f"  Sample size:              {len(bep_true_flows)} points")
    
    print(f"\nError Metrics:")
    print(f"  Mean Absolute Error (MAE):  {mae:.4f} m³/h")
    print(f"  Mean Absolute % Error:       {mape:.2f}%")
    print(f"  Root Mean Square Error:      {rmse:.4f} m³/h")
    
    # =========================================================================
    # SAVE STATISTICAL RESULTS TO JSON
    # =========================================================================
    statistics = {
        'correlation_analysis': {
            'pearson_r': float(pearson_r),
            'r_squared': float(r_squared),
            'p_value': float(pearson_p),
            'sample_size': len(bep_true_flows)
        },
        'error_metrics': {
            'mean_absolute_error_m3h': float(mae),
            'mean_absolute_percentage_error': float(mape),
            'root_mean_square_error_m3h': float(rmse)
        },
        'regression_coefficients': {
            'slope': float(coeffs[0]),
            'intercept': float(coeffs[1])
        },
        'bep_summary': {
            'heads_tested': target_heads,
            'true_bep_flows': bep_true_flows.tolist(),
            'proxy_bep_flows': bep_proxy_flows.tolist()
        }
    }
    
    json_path = output_dir / "statistical_results.json"
    with open(json_path, 'w') as f:
        json.dump(statistics, f, indent=4)
    print(f"\n✓ Statistical results saved: {json_path}")
    
    # =========================================================================
    # CREATE SUMMARY REPORT
    # =========================================================================
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE BEP DETECTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("TEST CONDITIONS:\n")
        f.write(f"  Heads tested: {target_heads} m\n")
        f.write(f"  Frequency range: 30-65 Hz\n")
        f.write(f"  Pump model: Schneider SUB 15-0.5cv\n\n")
        
        f.write("BEP DETECTION RESULTS:\n")
        f.write("-"*80 + "\n")
        for entry in bep_data:
            f.write(f"\nHead = {entry['head_m']}m:\n")
            f.write(f"  True BEP:  {entry['true_bep_flow_m3h']:.3f} m³/h ")
            f.write(f"(η = {entry['true_bep_efficiency']*100:.1f}%) @ {entry['true_bep_frequency_hz']:.1f} Hz\n")
            f.write(f"  Proxy BEP: {entry['proxy_bep_flow_m3h']:.3f} m³/h ")
            f.write(f"@ {entry['proxy_bep_frequency_hz']:.1f} Hz\n")
            f.write(f"  Error:     {entry['absolute_error_m3h']:.3f} m³/h ")
            f.write(f"({entry['percentage_error']:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL ANALYSIS:\n")
        f.write("="*80 + "\n\n")
        f.write(f"Correlation Metrics:\n")
        f.write(f"  Pearson r:  {pearson_r:.4f}\n")
        f.write(f"  R²:         {r_squared:.4f}\n")
        f.write(f"  p-value:    {pearson_p:.4f}\n\n")
        
        f.write(f"Error Metrics:\n")
        f.write(f"  MAE:        {mae:.4f} m³/h\n")
        f.write(f"  MAPE:       {mape:.2f}%\n")
        f.write(f"  RMSE:       {rmse:.4f} m³/h\n\n")
        
        f.write("INTERPRETATION:\n")
        if abs(pearson_r) > 0.9:
            f.write("  → Very strong linear correlation\n")
        elif abs(pearson_r) > 0.7:
            f.write("  → Strong linear correlation\n")
        elif abs(pearson_r) > 0.5:
            f.write("  → Moderate linear correlation\n")
        else:
            f.write("  → Weak linear correlation\n")
        
        if mape < 5:
            f.write("  → Excellent BEP detection accuracy\n")
        elif mape < 10:
            f.write("  → Good BEP detection accuracy\n")
        elif mape < 20:
            f.write("  → Acceptable BEP detection accuracy\n")
        else:
            f.write("  → Poor BEP detection accuracy\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Analysis report saved: {report_path}")
    
    print(f"\n{'='*80}")
    print(f"ALL RESULTS SAVED TO: {output_dir.absolute()}")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"  DATA FILES:")
    print(f"    • chart12_efficiency_curves_data.csv")
    print(f"    • chart3_frequency_flow_data.csv")
    print(f"    • chart4_bep_comparison_data.csv")
    print(f"    • statistical_results.json")
    print(f"    • analysis_report.txt")
    print(f"\n  CHART FILES (PNG + PDF):")
    print(f"    • chart1_true_efficiency")
    print(f"    • chart2_qp_proxy")
    print(f"    • chart3_frequency_vs_flow")
    print(f"    • chart4_bep_correlation")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    create_complete_analysis()