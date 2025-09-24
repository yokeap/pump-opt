# ==============================================================================
# FILE: main.py
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.pump_model import RealisticPumpSimulator
from src.optimizers import TPEOptimizer, ExtremumSeekingControl, RandomSearch
from src.proxy_functions import VolumetricEfficiencyProxy, NormalizedProxy
from src.visualization import PublicationPlotter
from src.experiments import ExperimentRunner
from src.utils import setup_publication_style

def main():
    """Main research execution"""
    
    print("🔬 BEP TRACKING RESEARCH PROJECT")
    print("=" * 50)
    
    # Setup
    setup_publication_style()
    
    # 1. Volumetric Efficiency Proxy Validation
    print("\n1️⃣ Volumetric Efficiency Proxy Validation")
    run_proxy_validation()
    
    # 2. Dynamic Head Change Testing
    print("\n2️⃣ Dynamic Head Change Testing") 
    run_dynamic_testing()
    
    # 3. Comprehensive Algorithm Comparison
    print("\n3️⃣ Algorithm Performance Comparison")
    run_algorithm_comparison()
    
    # # 4. Real-time Implementation Demo
    # print("\n4️⃣ Real-time Implementation Demo")
    # run_realtime_demo()
    # 4. UPDATED: Long-term Implementation Demo
    print("\n4️⃣ Long-term Implementation Demo (NOT real-time)")
    run_longterm_demo()
    
    print("\n🎉 Research Complete! All figures ready for publication.")

def run_proxy_validation():
    """Validate volumetric efficiency proxy performance"""
    runner = ExperimentRunner()
    plotter = PublicationPlotter()
    
    # Test proxy across different conditions
    results = runner.validate_proxy_function(
        # proxy_class=VolumetricEfficiencyProxy,
        proxy_class=NormalizedProxy,
        test_heads=[20, 25, 30, 35, 40, 45, 50],
        test_flows=[0.7, 0.8, 0.9, 1.0, 1.1],  # Relative to rated
        noise_levels=[0.01, 0.02, 0.03, 0.05]
    )
    
    # Generate publication figures
    plotter.plot_proxy_validation(results)

def run_dynamic_testing():
    """Test dynamic head changes during optimization with REALISTIC timescales"""
    runner = ExperimentRunner() 
    plotter = PublicationPlotter()
    
    print("Running REALISTIC dynamic head testing...")
    print("This simulates long-term pump operation with gradual system changes")
    
    results = runner.dynamic_head_test(
        initial_head=30,
        head_changes=[(50, 35), (70, 40)],  # Realistic: changes after TPE converges
        optimization_method='TPE',
        max_iterations=80,  # Increased for proper TPE convergence
        convergence_threshold=35  # TPE needs ~35 iterations to stabilize
    )
    
    # Print realistic interpretation
    print(f"\nRealistic Timeline Interpretation:")
    print(f"- Total duration: ~{80 * 10 / 60:.1f} hours of operation")
    print(f"- Head change 1: After ~{50 * 10 / 60:.1f} hours (gradual system change)")
    print(f"- Head change 2: After ~{70 * 10 / 60:.1f} hours (continued evolution)")
    print(f"- Application: Long-term efficiency optimization over days")
    
    plotter.plot_dynamic_performance(results)
    
    return results

def run_algorithm_comparison():
    """Compare TPE vs classical methods"""
    runner = ExperimentRunner()
    plotter = PublicationPlotter()
    
    results = runner.comprehensive_comparison(
        methods=['TPE', 'ESC', 'Random', 'GridSearch'],
        test_scenarios=[
            {'head': 30, 'noise': 0.05, 'name': 'Baseline'},
            {'head': 45, 'noise': 0.05, 'name': 'High Head'},  
            {'head': 25, 'noise': 0.075, 'name': 'Noisy'},
            {'head': 35, 'noise': 0.05, 'name': 'Medium Head'}
        ],
        iterations=25,
        n_trials=10  # Multiple runs for statistics
    )
    
    plotter.plot_algorithm_comparison(results)

# def run_realtime_demo():
#     """Demonstrate real-time implementation"""
#     runner = ExperimentRunner()
#     plotter = PublicationPlotter()
    
#     results = runner.realtime_demo(
#         duration_minutes=10,
#         update_interval_seconds=30,
#         disturbances=[
#             (3*60, 'head_change', 40),   # 3 min: head change
#             (6*60, 'noise_increase', 0.05), # 6 min: noise increase
#             (8*60, 'head_change', 30)    # 8 min: back to normal
#         ]
#     )
    
#     plotter.plot_realtime_demo(results)

def run_longterm_demo():
    """Demonstrate long-term optimization (renamed from realtime_demo)"""
    runner = ExperimentRunner()
    plotter = PublicationPlotter()
    
    print("Running LONG-TERM optimization demonstration...")
    print("This shows TPE performance over realistic operational timescales")
    
    results = runner.realtime_demo(  # Function name unchanged, but parameters realistic
        duration_hours=8.0,  # 8-hour demonstration
        update_interval_minutes=10.0,  # 10-minute updates (realistic for pumps)
        scenario_type='maintenance_optimization',
        disturbances=[
            (2*3600, 'head_drift', 32),      # 2 hours: gradual drift
            (5*3600, 'measurement_degradation', 0.03),  # 5 hours: sensor wear
            (6.5*3600, 'head_drift', 28)     # 6.5 hours: continued change
        ]
    )
    
    # Print realistic context
    print(f"\nLong-term Operation Context:")
    print(f"- Duration: 8 hours (typical work shift)")
    print(f"- Updates: Every 10 minutes (practical for pump systems)")
    print(f"- Scenarios: Gradual system changes, not rapid disturbances")
    print(f"- TPE convergence: ~5 hours for stable optimization")
    
    plotter.plot_realtime_demo(results)  # Visualization function name unchanged
    
    return results

def analyze_results(results):
    """Analyze and visualize results"""
    
    heads = [r['head'] for r in results]
    true_beps = [r['true_bep'] for r in results]
    predicted_beps = [r['predicted_bep'] for r in results]
    errors = [r['error'] for r in results]
    efficiencies = [r['best_efficiency'] for r in results]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TPE Head-Free BEP Tracking - Proof of Concept', fontsize=16)
    
    # Plot 1: True vs Predicted BEP
    ax1 = axes[0, 0]
    ax1.plot(heads, true_beps, 'ro-', label='True BEP', linewidth=2, markersize=8)
    ax1.plot(heads, predicted_beps, 'bs-', label='TPE Prediction', linewidth=2, markersize=8)
    ax1.fill_between(heads, 
                     np.array(true_beps) - 2, 
                     np.array(true_beps) + 2, 
                     alpha=0.2, color='red', label='±2 Hz Target')
    ax1.set_xlabel('System Head (m)')
    ax1.set_ylabel('BEP Frequency (Hz)')
    ax1.set_title('True vs Predicted BEP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error
    ax2 = axes[0, 1]
    bars = ax2.bar(heads, errors, color=['green' if e < 2 else 'orange' if e < 5 else 'red' for e in errors])
    ax2.axhline(y=2.0, color='green', linestyle='--', label='2 Hz Target')
    ax2.axhline(y=5.0, color='orange', linestyle='--', label='5 Hz Acceptable')
    ax2.set_xlabel('System Head (m)')
    ax2.set_ylabel('Prediction Error (Hz)')
    ax2.set_title('TPE Prediction Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best Efficiency Achieved
    ax3 = axes[1, 0]
    ax3.bar(heads, efficiencies, color='skyblue', alpha=0.7)
    ax3.set_xlabel('System Head (m)')
    ax3.set_ylabel('Best Efficiency Achieved')
    ax3.set_title('TPE Performance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    success_rate = sum(1 for e in errors if e < 2.0) / len(errors) * 100
    good_rate = sum(1 for e in errors if e < 5.0) / len(errors) * 100
    mean_efficiency = np.mean(efficiencies)
    
    summary_text = f"""
    PROOF OF CONCEPT SUMMARY
    ========================
    
    Mean Prediction Error: {mean_error:.2f} Hz
    Max Prediction Error:  {max_error:.2f} Hz
    
    Success Rate (< 2 Hz): {success_rate:.0f}%
    Good Rate (< 5 Hz):    {good_rate:.0f}%
    
    Mean Best Efficiency:  {mean_efficiency:.3f}
    
    CONCLUSION:
    {'✅ PROOF SUCCESSFUL!' if success_rate >= 60 else '⚠️  Needs Improvement'}
    
    TPE can find BEP without 
    knowing system head!
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("🎯 DETAILED PROOF OF CONCEPT RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"Head {r['head']:2d}m: True={r['true_bep']:5.1f}Hz, "
              f"Predicted={r['predicted_bep']:5.1f}Hz, Error={r['error']:4.1f}Hz, "
              f"Eff={r['best_efficiency']:.3f}")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Mean Error: {mean_error:.2f} Hz")
    print(f"   Success Rate (±2Hz): {success_rate:.0f}%")
    print(f"   Good Rate (±5Hz): {good_rate:.0f}%")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT PROOF OF CONCEPT!")
        print("   TPE reliably finds BEP without head information")
    elif success_rate >= 60:
        print("\n✅ GOOD PROOF OF CONCEPT!")
        print("   TPE successfully demonstrates head-free BEP finding")
    else:
        print("\n⚠️  PARTIAL PROOF OF CONCEPT")
        print("   Concept works but needs refinement")
    
    return {
        'mean_error': mean_error,
        'success_rate': success_rate,
        'good_rate': good_rate,
        'mean_efficiency': mean_efficiency
    }

def run_proof_of_concept():
    """Run simple proof of concept test"""
    
    print("🔬 PROOF OF CONCEPT: TPE Head-Free BEP Finding")
    print("=" * 55)
    
    # Test different system heads (TPE doesn't know these!)
    test_heads = [25, 30, 35, 40, 45]
    results = []
    
    for head in test_heads:
        print(f"\n📍 Testing System Head: {head}m (HIDDEN from TPE)")
        print("-" * 40)
        
        # Setup pump and optimizer
        pump = RealisticPumpSimulator(head)
        
        # Run optimization
        print("TPE Optimization Progress:")

        optimizer = TPEOptimizer(proxy_function=VolumetricEfficiencyProxy())
        runner = ExperimentRunner()
                
        trial_results = runner._run_single_optimization(
            pump, optimizer, max_iterations=15
        )
        
        true_bep, true_eff = pump.get_true_bep()
        best_freq, best_proxy = optimizer.get_best_bep()
        error = abs(best_freq - true_bep) if best_freq else float('inf')
        
        # Find best true efficiency achieved
        best_true_eff = max([h['true_efficiency'] for h in optimizer.history])
        
        result = {
            'head': head,
            'true_bep': true_bep,
            'predicted_bep': best_freq,
            'error': error,
            'best_efficiency': true_eff
        }
        results.append(result)
        
        print(f"\n📊 Results:")
        print(f"   True BEP: {true_bep:.1f} Hz")
        print(f"   TPE Prediction: {best_freq:.1f} Hz")
        print(f"   Error: {error:.1f} Hz")
        print(f"   Best Efficiency: {best_true_eff:.3f}")
        
        if error < 2.0:
            print("   ✅ SUCCESS: Within 2 Hz!")
        elif error < 5.0:
            print("   ✅ GOOD: Within 5 Hz")
        else:
            print("   ❌ NEEDS IMPROVEMENT")
    
    return results

if __name__ == "__main__":
    main()
    # results = run_proof_of_concept()
    # summary = analyze_results(results)