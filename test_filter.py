# ==============================================================================
# FILE: test_filter_effectiveness.py
# Test filtering effectiveness on noisy commercial pump data
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from src.commercial_pump_model import CommercialPumpSimulator
from src.proxy_functions_filtered import (
    firstOrderProxy, 
    compare_filters,
    create_proxy_with_noise_level,
    RECOMMENDED_FILTERS
)

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
})

def collect_noisy_data(pump, target_head, noise_level=0.02):
    """Collect data with noise"""
    frequencies = np.linspace(30, 65, 50)
    measurements = []
    
    for freq in frequencies:
        flow = pump.calculate_flow_from_frequency_head(freq, target_head, tolerance=0.1)
        
        if flow is None or flow < 0.1 or flow > pump.max_flow:
            continue
        
        # Calculate true values
        eff = pump._calculate_pump_efficiency(flow, freq)
        power, current, pf = pump._calculate_electrical_power(flow, freq)
        
        # Add noise to measurements
        flow_noisy = flow * (1 + np.random.normal(0, noise_level))
        power_noisy = power * (1 + np.random.normal(0, noise_level))
        pf_noisy = np.clip(pf + np.random.normal(0, noise_level*0.5), 0.6, 1.0)
        
        # Create measurement object
        class Measurement:
            def __init__(self, flow, power, pf, true_eff):
                self.flow = flow
                self.power = power
                self.power_factor = pf
                self.true_efficiency = true_eff
        
        measurements.append(Measurement(flow_noisy, power_noisy, pf_noisy, eff))
    
    return measurements

def test_all_filters():
    """Test all filter types on noisy data"""
    
    output_dir = Path("results/debug_commercial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FILTER EFFECTIVENESS TEST")
    print("="*80)
    
    # Test different noise levels
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    target_head = 30.0
    
    for noise_level in noise_levels:
        print(f"\n{'='*80}")
        print(f"NOISE LEVEL: {noise_level*100:.1f}%")
        print(f"{'='*80}")
        
        # Initialize pump with noise
        pump = CommercialPumpSimulator(system_head=target_head, noise_level=0.0)
        
        # Collect measurements
        measurements = collect_noisy_data(pump, target_head, noise_level)
        
        flows = np.array([m.flow for m in measurements])
        true_effs = np.array([m.true_efficiency for m in measurements])
        
        # Find true BEP
        true_bep_idx = np.argmax(true_effs)
        true_bep_flow = flows[true_bep_idx]
        
        print(f"\nTrue BEP Flow: {true_bep_flow:.3f} m³/h")
        print(f"Number of measurements: {len(measurements)}")
        
        # Test each filter type
        filters = ['none', 'savgol', 'gaussian', 'median', 'moving_average']
        results = {}
        
        for filter_type in filters:
            if filter_type == 'none':
                proxy = firstOrderProxy(filter_type='none')
            else:
                proxy = create_proxy_with_noise_level(firstOrderProxy, 
                                                     'high_noise' if noise_level > 0.03 else 'medium_noise')
                # Override to test specific filter
                proxy.filter_type = filter_type
                
                if filter_type == 'savgol':
                    proxy.filter_params = {'window_length': 11, 'polyorder': 3}
                elif filter_type == 'gaussian':
                    proxy.filter_params = {'sigma': 2.5}
                elif filter_type == 'median':
                    proxy.filter_params = {'kernel_size': 7}
                elif filter_type == 'moving_average':
                    proxy.filter_params = {'window': 9}
            
            # Calculate proxy values
            raw_values, filtered_values = proxy.calculate_batch(measurements)
            
            # Find BEP from filtered data
            bep_idx = np.argmax(filtered_values)
            bep_flow = flows[bep_idx]
            
            # Calculate error
            error = abs(bep_flow - true_bep_flow)
            error_pct = (error / true_bep_flow) * 100
            
            results[filter_type] = {
                'bep_flow': bep_flow,
                'error': error,
                'error_pct': error_pct,
                'raw_values': raw_values,
                'filtered_values': filtered_values
            }
            
            print(f"\n{filter_type.upper():15s}: BEP = {bep_flow:.3f} m³/h, Error = {error_pct:.2f}%")
        
        # Find best filter
        best_filter = min(results.items(), key=lambda x: x[1]['error_pct'])
        print(f"\n✓ BEST FILTER: {best_filter[0].upper()} (Error: {best_filter[1]['error_pct']:.2f}%)")
        
        # Plot comparison
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Filter Comparison - Noise Level: {noise_level*100:.1f}%', 
                    fontsize=14, fontweight='bold')
        
        for idx, (filter_type, ax) in enumerate(zip(filters, axes.flat)):
            if idx >= len(filters):
                ax.axis('off')
                continue
                
            data = results[filter_type]
            
            # Plot raw and filtered
            ax.plot(flows, data['raw_values'], 'o', alpha=0.3, 
                   markersize=4, color='gray', label='Raw')
            ax.plot(flows, data['filtered_values'], 'b-', linewidth=2.5, 
                   label='Filtered')
            
            # Mark BEPs
            ax.axvline(data['bep_flow'], color='b', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f"Proxy BEP: {data['bep_flow']:.3f}")
            ax.axvline(true_bep_flow, color='r', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f"True BEP: {true_bep_flow:.3f}")
            
            # Styling
            ax.set_xlabel('Flow (m³/h)', fontweight='bold')
            ax.set_ylabel('Proxy Value', fontweight='bold')
            ax.set_title(f'{filter_type.upper()} - Error: {data["error_pct"]:.2f}%', 
                        fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / f"filter_comparison_noise_{noise_level*100:.0f}pct.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved: {fig_path}")
        
        plt.close()
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: RECOMMENDED FILTERS BY NOISE LEVEL")
    print(f"{'='*80}")
    print("\nNoise Level  | Recommended Filter      | Window/Params")
    print("-"*80)
    print("0-2%         | none or savgol          | window=7, poly=2")
    print("2-5%         | savgol                  | window=9, poly=3")
    print("5-10%        | savgol or median        | window=11, poly=3")
    print("10%+         | median or gaussian      | kernel=7 or sigma=2.5")
    print("-"*80)
    print("\nFor outliers or spikes: Use median filter")
    print("For smooth data: Use savgol filter")
    print("For aggressive smoothing: Use gaussian filter")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_all_filters()