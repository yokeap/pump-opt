# ==============================================================================
# FILE: debug_proxy_commercial_filtered.py
# Publication-Quality BEP Detection Analysis with Savitzky-Golay Filtering
# Origin Software Style for Academic Publication
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions import firstOrderProxy
import json
import pandas as pd
from pathlib import Path

rcParams.update({
    # --- Font and text ---
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

    # --- Grid style ---
    'axes.grid': False,
})

def apply_savgol_filter(data: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth noisy data
    
    Args:
        data: Input data array
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial to fit
        
    Returns:
        Filtered data
    """
    if len(data) < window_length:
        # Not enough data points, reduce window
        window_length = len(data)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(5, window_length)
        polyorder = min(polyorder, window_length - 1)
    
    return savgol_filter(data, window_length, polyorder)

def collect_data_at_constant_head(pump, proxy, target_head, freq_range=(30, 65, 100)):
    """Collect data at a single constant head"""
    frequencies = np.linspace(*freq_range)
    
    flows, true_effs, proxy_values, valid_freqs = [], [], [], []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.1)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        # Calculate efficiency and power at THIS SPECIFIC FLOW
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        # Create a simple measurement object for proxy calculation
        class SimpleMeasurement:
            def __init__(self, flow, power, power_factor):
                self.flow = flow
                self.power = power
                self.power_factor = power_factor
        
        measurement = SimpleMeasurement(flow, power, pf)
        
        # Calculate proxy
        proxy_value = proxy.calculate(measurement)
        
        flows.append(flow)
        true_effs.append(eff)
        proxy_values.append(proxy_value)
        valid_freqs.append(freq)
    
    return np.array(flows), np.array(true_effs), np.array(proxy_values), np.array(valid_freqs)

def create_publication_charts(noise_level=0.02, filter_window=9, filter_poly=3):
    """
    Create four publication-quality charts with Savitzky-Golay filtering:
    1. True Efficiency vs Flow (multiple heads)
    2. Q/P Proxy vs Flow (multiple heads) - WITH FILTERING
    3. Frequency vs Flow (multiple heads)
    4. BEP Correlation (True vs Proxy)
    
    All results saved to: results/debug_commercial/
    """
    
    # Create output directory
    output_dir = Path("results/debug_commercial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PUBLICATION-QUALITY BEP DETECTION ANALYSIS")
    print(f"Noise Level: {noise_level*100:.1f}%")
    print(f"Savitzky-Golay Filter: window={filter_window}, polyorder={filter_poly}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Initialize
    pump = CommercialPumpSimulator(system_head=30.0, noise_level=noise_level)
    proxy = firstOrderProxy()
    
    # Test heads
    target_heads = [25, 30, 35, 40]
    colors = ['#0072BD', '#D95319', '#77AC30', '#EDB120']
    markers = ['o', 's', '^', 'd']
    
    # Collect all data
    all_data = {}
    all_raw_data = []
    
    for head in target_heads:
        flows, effs, proxies_raw, freqs = collect_data_at_constant_head(pump, proxy, head)
        
        # Apply Savitzky-Golay filter to proxy values
        proxies_filtered = apply_savgol_filter(proxies_raw, filter_window, filter_poly)
        
        # Find BEP for each head
        bep_idx = np.argmax(effs)
        proxy_bep_idx = np.argmax(proxies_filtered)  # Use FILTERED data for BEP detection
        
        all_data[head] = {
            'flows': flows,
            'effs': effs,
            'proxies_raw': proxies_raw,
            'proxies_filtered': proxies_filtered,
            'freqs': freqs,
            'bep_flow': flows[bep_idx],
            'bep_eff': effs[bep_idx],
            'proxy_bep_flow': flows[proxy_bep_idx],
            'proxy_bep_value': proxies_filtered[proxy_bep_idx]
        }
        
        # Store for CSV
        for j in range(len(flows)):
            all_raw_data.append({
                'head_m': head,
                'flow_m3h': flows[j],
                'frequency_hz': freqs[j],
                'true_efficiency': effs[j],
                'proxy_value_raw': proxies_raw[j],
                'proxy_value_filtered': proxies_filtered[j]
            })
        
        error = abs(flows[proxy_bep_idx] - flows[bep_idx])
        error_pct = (error / flows[bep_idx]) * 100
        
        print(f"\nHead = {head}m:")
        print(f"  True BEP:  {flows[bep_idx]:.3f} m³/h (η={effs[bep_idx]*100:.1f}%)")
        print(f"  Proxy BEP: {flows[proxy_bep_idx]:.3f} m³/h (filtered)")
        print(f"  Error:     {error:.3f} m³/h ({error_pct:.1f}%)")
    
    # Save raw data
    df_raw = pd.DataFrame(all_raw_data)
    csv_path = output_dir / "chart12_efficiency_curves_data.csv"
    df_raw.to_csv(csv_path, index=False)
    print(f"\n✓ Chart 1&2 data saved: {csv_path}")
    
    # =========================================================================
    # CHART 1: TRUE EFFICIENCY
    # =========================================================================
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    
    for i, head in enumerate(target_heads):
        data = all_data[head]
        
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
    ax1.set_title('True Pump Efficiency at Constant Head', 
                 fontsize=14, fontweight='bold', pad=15)
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
    # CHART 2: Q/P PROXY WITH FILTERING
    # =========================================================================
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    for i, head in enumerate(target_heads):
        data = all_data[head]
        
        # Plot RAW proxy values (light/transparent)
        ax2.plot(data['flows'], data['proxies_raw'], 'o', 
                color=colors[i], alpha=0.2, markersize=4, zorder=1)
        
        # Plot FILTERED proxy curve (bold)
        ax2.plot(data['flows'], data['proxies_filtered'], 
                color=colors[i], linewidth=2.5, 
                label=f'{head} m', zorder=2)
        
        # Mark proxy BEP with cross (on filtered data)
        ax2.plot(data['proxy_bep_flow'], data['proxy_bep_value'], 
                marker='+', color=colors[i], markersize=14, 
                markeredgewidth=3, zorder=3)
    
    ax2.set_xlabel('Flow (m³/h)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Q/P Proxy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Q/P Proxy with Savitzky-Golay Filter (w={filter_window}, p={filter_poly})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(title='Head', loc='upper right', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, np.max([all_data[h]['proxies_filtered'].max() for h in target_heads])*1.1)
    ax2.grid(False)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
    ax2.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, top=True, right=True)
    
    plt.tight_layout()
    fig2_png = output_dir / "chart2_qp_proxy_filtered.png"
    fig2_pdf = output_dir / "chart2_qp_proxy_filtered.pdf"
    plt.savefig(fig2_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig2_pdf, bbox_inches='tight')
    print(f"✓ Chart 2 saved: {fig2_png}")
    print(f"✓ Chart 2 saved: {fig2_pdf}")
    
    # =========================================================================
    # CHART 3: FREQUENCY vs FLOW
    # =========================================================================
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    
    chart3_raw_data = []
    
    for i, head in enumerate(target_heads):
        data = all_data[head]
        
        # Store for CSV
        for j in range(len(data['freqs'])):
            chart3_raw_data.append({
                'head_m': head,
                'frequency_hz': data['freqs'][j],
                'flow_m3h': data['flows'][j],
                'true_efficiency': data['effs'][j],
                'proxy_value_filtered': data['proxies_filtered'][j]
            })
        
        # Plot frequency vs flow
        ax3.plot(data['freqs'], data['flows'], color=colors[i], linewidth=2.5, 
                marker=markers[i], markersize=5, markevery=5,
                label=f'{head} m', zorder=2)
    
    # Save Chart 3 data
    df_ch3 = pd.DataFrame(chart3_raw_data)
    csv_ch3 = output_dir / "chart3_frequency_flow_data.csv"
    df_ch3.to_csv(csv_ch3, index=False)
    print(f"✓ Chart 3 data saved: {csv_ch3}")
    
    ax3.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Flow (m³/h)', fontsize=13, fontweight='bold')
    ax3.set_title('Flow vs Frequency at Constant Head', 
                 fontsize=14, fontweight='bold', pad=15)
    ax3.legend(title='Head', loc='upper left', frameon=True, 
              edgecolor='black', fancybox=False, fontsize=11)
    ax3.set_xlim(25, 70)
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
    print(f"✓ Chart 3 saved: {fig3_png}")
    print(f"✓ Chart 3 saved: {fig3_pdf}")
    
    # =========================================================================
    # CHART 4: BEP CORRELATION
    # =========================================================================
    fig4 = plt.figure(figsize=(8, 8))
    ax4 = fig4.add_subplot(111)
    
    bep_true_flows = []
    bep_proxy_flows = []
    bep_data = []
    
    print(f"\n{'='*80}")
    print("BEP DETECTION COMPARISON (FILTERED):")
    print(f"{'='*80}")
    
    for i, head in enumerate(target_heads):
        data = all_data[head]
        
        bep_true_flow = data['bep_flow']
        bep_true_eff = data['bep_eff']
        bep_proxy_flow = data['proxy_bep_flow']
        
        bep_true_flows.append(bep_true_flow)
        bep_proxy_flows.append(bep_proxy_flow)
        
        error = abs(bep_proxy_flow - bep_true_flow)
        error_pct = (error / bep_true_flow) * 100
        
        bep_data.append({
            'head_m': head,
            'true_bep_flow_m3h': bep_true_flow,
            'true_bep_efficiency': bep_true_eff,
            'proxy_bep_flow_m3h': bep_proxy_flow,
            'absolute_error_m3h': error,
            'percentage_error': error_pct
        })
        
        print(f"\nHead = {head}m:")
        print(f"  True BEP:  Q = {bep_true_flow:.3f} m³/h (η = {bep_true_eff*100:.1f}%)")
        print(f"  Proxy BEP: Q = {bep_proxy_flow:.3f} m³/h (filtered)")
        print(f"  Error:     {error:.3f} m³/h ({error_pct:.1f}%)")
        
        # Plot BEP point
        ax4.scatter(bep_true_flow, bep_proxy_flow, 
                   color=colors[i], marker=markers[i], s=200, 
                   alpha=0.7, edgecolors='black', linewidth=2,
                   label=f'{head} m', zorder=2)
    
    # Save BEP data
    df_bep = pd.DataFrame(bep_data)
    bep_csv = output_dir / "chart4_bep_comparison_data.csv"
    df_bep.to_csv(bep_csv, index=False)
    print(f"\n✓ Chart 4 data saved: {bep_csv}")
    
    bep_true_flows = np.array(bep_true_flows)
    bep_proxy_flows = np.array(bep_proxy_flows)
    
    # Calculate correlation
    pearson_r, pearson_p = pearsonr(bep_true_flows, bep_proxy_flows)
    r_squared = r2_score(bep_true_flows, bep_proxy_flows)
    
    # Fit regression
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
    
    ax4.set_xlabel('True BEP Flow (m³/h)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Proxy BEP Flow (m³/h)', fontsize=13, fontweight='bold')
    ax4.set_title('BEP Detection: True Efficiency vs Filtered Q/P Proxy', 
                 fontsize=14, fontweight='bold', pad=15)
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
    print(f"✓ Chart 4 saved: {fig4_png}")
    print(f"✓ Chart 4 saved: {fig4_pdf}")
    
    plt.show()
    
    # =========================================================================
    # SAVE STATISTICS
    # =========================================================================
    mae = np.mean(np.abs(bep_proxy_flows - bep_true_flows))
    mape = np.mean(np.abs((bep_proxy_flows - bep_true_flows) / bep_true_flows)) * 100
    rmse = np.sqrt(np.mean((bep_proxy_flows - bep_true_flows)**2))
    
    statistics = {
        'filter_settings': {
            'noise_level': noise_level,
            'filter_type': 'savitzky_golay',
            'window_length': filter_window,
            'polyorder': filter_poly
        },
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
        'bep_summary': {
            'heads_tested': target_heads,
            'true_bep_flows': bep_true_flows.tolist(),
            'proxy_bep_flows': bep_proxy_flows.tolist()
        }
    }
    
    json_path = output_dir / "statistical_results_filtered.json"
    with open(json_path, 'w') as f:
        json.dump(statistics, f, indent=4)
    print(f"\n✓ Statistical results saved: {json_path}")
    
    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY:")
    print(f"{'='*80}")
    print(f"\nCorrelation:")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  R²:        {r_squared:.4f}")
    print(f"  p-value:   {pearson_p:.4f}")
    print(f"\nError Metrics:")
    print(f"  MAE:       {mae:.4f} m³/h")
    print(f"  MAPE:      {mape:.2f}%")
    print(f"  RMSE:      {rmse:.4f} m³/h")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run with different noise levels
    create_publication_charts(
        noise_level=0.01,     # 2% noise
        filter_window=9,      # Savitzky-Golay window
        filter_poly=3         # Polynomial order
    )