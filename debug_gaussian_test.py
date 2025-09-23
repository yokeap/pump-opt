# ==============================================================================
# Test BEP finding using the existing system
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.pump_model import RealisticPumpSimulator
from src.optimizers import TPEOptimizer, ExtremumSeekingControl
from src.proxy_functions import VolumetricEfficiencyProxy

def test_simple_bep_finding():
    """Test BEP finding using the current system"""
    
    print("Testing BEP finding")
    print("=" * 20)
    
    # Set up pump
    pump = RealisticPumpSimulator(
        system_head=35.0,      # head 35 meters
        rated_flow=1000.0,     # flow 1000 m³/h
        noise_level=0.02       # noise 2%
    )
    
    true_bep, true_eff = pump.get_true_bep()
    print(f"True BEP: {true_bep:.1f} Hz, True Efficiency: {true_eff:.3f}")
    
    # Set up TPE with Q/√P proxy
    proxy = VolumetricEfficiencyProxy()
    optimizer = TPEOptimizer(proxy_function=proxy)
    # optimizer = ExtremumSeekingControl(proxy_function=proxy)
    
    print(f"\nStarting optimization with TPE...")
    
    # Store results
    history = []
    
    for iteration in range(1, 26):  # 25 iterations
        # TPE suggests a frequency
        frequency = optimizer.suggest_frequency()
        
        # Measure pump response
        measurement = pump.get_measurement(frequency)
        
        # Update TPE
        optimizer.update(frequency, measurement)
        
        # Calculate proxy value = Q/√P
        proxy_value = proxy.calculate(measurement)
        
        # Get current best BEP
        best_freq, best_proxy = optimizer.get_best_bep()
        error = abs(best_freq - true_bep) if best_freq else float('inf')
        
        # Store results
        history.append({
            'iteration': iteration,
            'frequency': frequency,
            'Q': measurement.flow,
            'P': measurement.power,
            'proxy': proxy_value,
            'true_efficiency': measurement.true_efficiency,
            'best_frequency': best_freq,
            'error': error
        })
        
        # Display progress every 5 iterations or first 3 iterations
        if iteration % 5 == 0 or iteration <= 3:
            print(f"Iteration {iteration:2d}: Frequency {frequency:5.1f} Hz, Q={measurement.flow:6.1f}, P={measurement.power:5.1f}, "
                  f"Proxy={proxy_value:6.3f}, Current BEP={best_freq:5.1f} Hz, Error={error:4.1f} Hz")
    
    # Analyze results
    final_error = history[-1]['error']
    mean_error = np.mean([h['error'] for h in history[-5:]])  # mean of last 5 iterations
    
    print(f"\nFinal results:")
    print(f"True BEP: {true_bep:.1f} Hz")
    print(f"Found BEP: {best_freq:.1f} Hz")
    print(f"Error: {final_error:.2f} Hz")
    print(f"Mean error (last 5 iterations): {mean_error:.2f} Hz")
    
    if final_error < 2.0:
        print("Result: Success (error < 2 Hz)")
    elif final_error < 5.0:
        print("Result: Acceptable (error < 5 Hz)")
    else:
        print("Result: Needs improvement (error > 5 Hz)")
    
    # Plot results
    plot_results(history, true_bep)
    
    return history

def plot_results(history, true_bep):
    """Plot BEP finding results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('BEP Finding Results using TPE + Proxy Q/√P', fontsize=14)
    
    iterations = [h['iteration'] for h in history]
    frequencies = [h['frequency'] for h in history]
    proxy_values = [h['proxy'] for h in history]
    errors = [h['error'] for h in history]
    best_frequencies = [h['best_frequency'] for h in history]
    
    # 1. Tested frequency vs true BEP
    ax1 = axes[0, 0]
    ax1.scatter(iterations, frequencies, alpha=0.7, s=30, label='Tested Frequency')
    ax1.axhline(true_bep, color='red', linestyle='--', label=f'True BEP ({true_bep:.1f} Hz)')
    ax1.plot(iterations, best_frequencies, 'g-', linewidth=2, alpha=0.7, label='Current BEP')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('(a) Convergence to BEP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Proxy values
    ax2 = axes[0, 1]
    ax2.plot(iterations, proxy_values, 'b-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Proxy Value (Q/√P)')
    ax2.set_title('(b) Proxy value per iteration')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error
    ax3 = axes[1, 0]
    ax3.plot(iterations, errors, 'r-o', markersize=4, linewidth=1.5)
    ax3.axhline(2.0, color='green', linestyle='--', alpha=0.7, label='Target (2 Hz)')
    ax3.axhline(5.0, color='orange', linestyle='--', alpha=0.7, label='Acceptable (5 Hz)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error (Hz)')
    ax3.set_title('(c) BEP prediction error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Q-P Curve
    ax4 = axes[1, 1]
    Q_values = [h['Q'] for h in history]
    P_values = [h['P'] for h in history]
    colors = proxy_values  # use proxy value as color
    
    scatter = ax4.scatter(Q_values, P_values, c=colors, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Flow Rate (m³/h)')
    ax4.set_ylabel('Power (kW)')
    ax4.set_title('(d) Q-P Points (color shows Proxy)')
    plt.colorbar(scatter, ax=ax4, label='Proxy Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bep_finding_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_proxy_methods():
    """Compare different proxy methods"""
    
    print("\nComparing Proxy Methods")
    print("=" * 25)
    
    # Set up pump
    pump = RealisticPumpSimulator(system_head=35.0, noise_level=0.02)
    true_bep, _ = pump.get_true_bep()
    
    # Import proxy functions
    from src.proxy_functions import VolumetricEfficiencyProxy, OriginalProxy, NormalizedProxy
    
    proxy_methods = {
        'Volumetric (Q/√P × PF)': VolumetricEfficiencyProxy(),
        'Original (Q²/P × PF)': OriginalProxy(),
        'Normalized (Q/√P × PF)': NormalizedProxy()
    }
    
    results = {}
    
    for method_name, proxy in proxy_methods.items():
        print(f"\nTesting {method_name}...")
        
        # Set up TPE
        optimizer = TPEOptimizer(proxy_function=proxy)
        
        # Run optimization
        for iteration in range(1, 21):  # 20 iterations
            frequency = optimizer.suggest_frequency()
            measurement = pump.get_measurement(frequency)
            optimizer.update(frequency, measurement)
        
        # Results
        best_freq, _ = optimizer.get_best_bep()
        error = abs(best_freq - true_bep) if best_freq else float('inf')
        
        results[method_name] = {
            'best_frequency': best_freq,
            'error': error,
            'history': optimizer.history
        }
        
        print(f"  Found BEP: {best_freq:.1f} Hz, Error: {error:.2f} Hz")
    
    # Summary
    print(f"\nComparison Summary:")
    print(f"True BEP: {true_bep:.1f} Hz")
    print("-" * 40)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['error'])
    
    for i, (method, result) in enumerate(sorted_results):
        status = "Success" if result['error'] < 2.0 else "Acceptable" if result['error'] < 5.0 else "Needs Improvement"
        print(f"{i+1}. {method}")
        print(f"   BEP: {result['best_frequency']:.1f} Hz, Error: {result['error']:.2f} Hz ({status})")
    
    return results

if __name__ == "__main__":
    # Simple BEP test
    history = test_simple_bep_finding()
    
    # Compare proxy methods
    comparison = compare_proxy_methods()
    
    print(f"\nTesting complete! Check the file bep_finding_results.png")
