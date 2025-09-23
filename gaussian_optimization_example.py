# ==============================================================================
# FILE: gaussian_optimization_example.py
# Complete example integrating Gaussian proxy with your existing system
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import your existing modules (adjust paths as needed)
from src.pump_model import RealisticPumpSimulator
from src.optimizers import TPEOptimizer
from src.experiments import ExperimentRunner
from src.utils import setup_publication_style
from src.visualization import PublicationPlotter

# Import new Gaussian proxy
from src.proxy_functions_gaussian import DataDrivenGaussianProxy, GaussianProxyTrainer

def run_gaussian_proxy_experiment():
    """
    Complete experiment comparing Gaussian proxy vs existing methods
    """
    
    print("🎯 GAUSSIAN PROXY OPTIMIZATION EXPERIMENT")
    print("=" * 50)
    
    # Setup
    setup_publication_style()
    
    # 1. Initialize pump simulator
    pump = RealisticPumpSimulator(
        system_head=35.0,
        rated_power=200.0,
        rated_flow=1000.0,
        noise_level=0.025
    )
    
    print(f"Pump setup: {pump.current_head}m head, {pump.rated_flow} m³/h rated flow")
    true_bep, true_eff = pump.get_true_bep()
    print(f"True BEP: {true_bep:.1f} Hz, Efficiency: {true_eff:.3f}")
    
    # 2. Train Gaussian proxy
    print(f"\n🔧 PHASE 1: GAUSSIAN PROXY TRAINING")
    print("-" * 35)
    
    trainer = GaussianProxyTrainer(pump)
    
    # Generate comprehensive training sweep
    training_measurements = trainer.generate_training_sweep(
        frequency_range=(25, 60),
        n_points=35,  # Dense sampling for good training
        noise_level=0.01  # Low noise for training
    )
    
    # Train multiple Gaussian variants
    trained_proxies = trainer.train_multiple_proxies(
        proxy_methods=['volumetric', 'original', 'power'],
        frequency_range=(25, 60),
        n_points=35,
        noise_level=0.01
    )
    
    # Compare and select best proxy
    comparison = trainer.compare_proxy_performance(trained_proxies)
    best_proxy_name = comparison['rankings']['by_correlation'][0][0]
    best_gaussian_proxy = trained_proxies[best_proxy_name]['proxy']
    
    print(f"\n✅ Selected best proxy: {best_proxy_name}")
    print(f"   Correlation with efficiency: {comparison['proxy_performance'][best_proxy_name]['correlation']:.3f}")
    print(f"   BEP prediction error: {comparison['proxy_performance'][best_proxy_name]['bep_error']:.2f} Hz")
    
    # 3. Run optimization comparison
    print(f"\n⚡ PHASE 2: OPTIMIZATION COMPARISON")
    print("-" * 40)
    
    results = {}
    
    # Test scenarios with different noise levels
    test_scenarios = [
        {'head': 30, 'noise': 0.02, 'name': 'Baseline'},
        {'head': 40, 'noise': 0.02, 'name': 'High Head'},
        {'head': 35, 'noise': 0.04, 'name': 'High Noise'}
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['name']} (Head: {scenario['head']}m, Noise: {scenario['noise']})")
        
        scenario_results = {}
        
        # Test with different proxy functions
        proxies_to_test = {
            'gaussian_best': best_gaussian_proxy,
            'volumetric_original': VolumetricEfficiencyProxy(),
            'original_q2p': OriginalProxy()
        }
        
        for proxy_name, proxy_func in proxies_to_test.items():
            print(f"  Testing with {proxy_name}...")
            
            # Setup pump for this scenario
            test_pump = RealisticPumpSimulator(
                system_head=scenario['head'],
                rated_flow=1000.0,
                noise_level=scenario['noise']
            )
            
            # Setup TPE optimizer
            optimizer = TPEOptimizer(proxy_function=proxy_func)
            
            # Run optimization
            start_time = time.time()
            trial_results = []
            
            for iteration in range(1, 26):  # 25 iterations
                frequency = optimizer.suggest_frequency()
                measurement = test_pump.get_measurement(frequency)
                optimizer.update(frequency, measurement)
                
                # Track progress
                true_bep_scenario, _ = test_pump.get_true_bep()
                best_freq, _ = optimizer.get_best_bep()
                error = abs(best_freq - true_bep_scenario) if best_freq else float('inf')
                
                trial_results.append({
                    'iteration': iteration,
                    'frequency': frequency,
                    'true_bep': true_bep_scenario,
                    'predicted_bep': best_freq,
                    'error': error,
                    'proxy_value': proxy_func.calculate(measurement),
                    'true_efficiency': measurement.true_efficiency
                })
            
            duration = time.time() - start_time
            
            # Calculate performance metrics
            final_error = trial_results[-1]['error']
            mean_error = np.mean([r['error'] for r in trial_results if r['error'] != float('inf')])
            success_rate = sum(1 for r in trial_results if r['error'] < 2.0) / len(trial_results)
            
            scenario_results[proxy_name] = {
                'final_error': final_error,
                'mean_error': mean_error,
                'success_rate': success_rate,
                'duration': duration,
                'history': trial_results
            }
            
            print(f"    Final error: {final_error:.2f} Hz, Success rate: {success_rate*100:.1f}%")
        
        results[scenario['name']] = scenario_results
    
    # 4. Dynamic head test with Gaussian proxy
    print(f"\n🔄 PHASE 3: DYNAMIC HEAD ADAPTATION")
    print("-" * 38)
    
    # Setup dynamic test
    dynamic_pump = RealisticPumpSimulator(system_head=30.0, noise_level=0.025)
    dynamic_optimizer = TPEOptimizer(proxy_function=best_gaussian_proxy)
    
    dynamic_results = []
    head_changes = [(15, 40), (25, 28)]  # Iteration, new_head
    
    print("Running dynamic adaptation test...")
    
    for iteration in range(1, 36):
        # Check for head changes
        for change_iter, new_head in head_changes:
            if iteration == change_iter:
                old_head = dynamic_pump.current_head
                dynamic_pump.set_system_head(new_head)
                print(f"  Iteration {iteration}: Head changed {old_head} -> {new_head} m")
        
        # Optimization step
        frequency = dynamic_optimizer.suggest_frequency()
        measurement = dynamic_pump.get_measurement(frequency)
        dynamic_optimizer.update(frequency, measurement)
        
        # Track results
        true_bep_dyn, _ = dynamic_pump.get_true_bep()
        best_freq_dyn, _ = dynamic_optimizer.get_best_bep()
        error_dyn = abs(best_freq_dyn - true_bep_dyn) if best_freq_dyn else float('inf')
        
        dynamic_results.append({
            'iteration': iteration,
            'frequency': frequency,
            'system_head': dynamic_pump.current_head,
            'true_bep': true_bep_dyn,
            'predicted_bep': best_freq_dyn,
            'error': error_dyn
        })
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: Error = {error_dyn:.2f} Hz")
    
    # 5. Results analysis and visualization
    print(f"\n📊 PHASE 4: RESULTS ANALYSIS")
    print("-" * 30)
    
    # Create comprehensive comparison
    comparison_summary = analyze_gaussian_results(results, dynamic_results, 
                                                best_gaussian_proxy, comparison)
    
    # Generate visualization
    create_gaussian_results_plots(results, dynamic_results, best_gaussian_proxy, 
                                trained_proxies, comparison)
    
    return {
        'trained_proxies': trained_proxies,
        'optimization_results': results,
        'dynamic_results': dynamic_results,
        'best_proxy': best_gaussian_proxy,
        'comparison_summary': comparison_summary,
        'proxy_comparison': comparison
    }

def analyze_gaussian_results(optimization_results, dynamic_results, best_proxy, proxy_comparison):
    """Analyze and summarize Gaussian proxy experiment results"""
    
    print("Analyzing experiment results...")
    
    summary = {
        'gaussian_performance': {},
        'comparison_with_existing': {},
        'dynamic_adaptation': {},
        'overall_assessment': {}
    }
    
    # 1. Gaussian proxy performance across scenarios
    gaussian_errors = []
    gaussian_success_rates = []
    
    for scenario_name, scenario_results in optimization_results.items():
        if 'gaussian_best' in scenario_results:
            gaussian_result = scenario_results['gaussian_best']
            gaussian_errors.append(gaussian_result['mean_error'])
            gaussian_success_rates.append(gaussian_result['success_rate'])
    
    summary['gaussian_performance'] = {
        'mean_error_across_scenarios': np.mean(gaussian_errors),
        'std_error_across_scenarios': np.std(gaussian_errors),
        'mean_success_rate': np.mean(gaussian_success_rates),
        'consistent_performance': np.std(gaussian_errors) < 1.0
    }
    
    # 2. Comparison with existing methods
    method_performance = {}
    
    for scenario_name, scenario_results in optimization_results.items():
        for method_name, method_result in scenario_results.items():
            if method_name not in method_performance:
                method_performance[method_name] = []
            method_performance[method_name].append(method_result['mean_error'])
    
    # Calculate improvement
    if 'gaussian_best' in method_performance and 'volumetric_original' in method_performance:
        gaussian_avg = np.mean(method_performance['gaussian_best'])
        volumetric_avg = np.mean(method_performance['volumetric_original'])
        improvement = ((volumetric_avg - gaussian_avg) / volumetric_avg) * 100
        
        summary['comparison_with_existing'] = {
            'gaussian_mean_error': gaussian_avg,
            'volumetric_mean_error': volumetric_avg,
            'improvement_percentage': improvement,
            'better_performance': improvement > 0
        }
    
    # 3. Dynamic adaptation analysis
    if dynamic_results:
        adaptation_errors = [r['error'] for r in dynamic_results if r['error'] != float('inf')]
        final_error = adaptation_errors[-1] if adaptation_errors else float('inf')
        
        # Find adaptation times after head changes
        head_change_iterations = [15, 25]
        adaptation_times = []
        
        for change_iter in head_change_iterations:
            post_change_errors = [r['error'] for r in dynamic_results 
                                if r['iteration'] > change_iter and r['error'] < 2.0]
            if post_change_errors:
                # Find first good result after change
                for r in dynamic_results:
                    if r['iteration'] > change_iter and r['error'] < 2.0:
                        adaptation_times.append(r['iteration'] - change_iter)
                        break
        
        summary['dynamic_adaptation'] = {
            'final_error': final_error,
            'mean_adaptation_time': np.mean(adaptation_times) if adaptation_times else float('inf'),
            'successful_adaptations': len(adaptation_times),
            'total_head_changes': len(head_change_iterations)
        }
    
    # 4. Overall assessment
    gaussian_perf = summary['gaussian_performance']
    comparison_perf = summary['comparison_with_existing']
    dynamic_perf = summary['dynamic_adaptation']
    
    # Scoring criteria
    accuracy_good = gaussian_perf['mean_error_across_scenarios'] < 2.0
    consistency_good = gaussian_perf['consistent_performance']
    improvement_good = comparison_perf.get('improvement_percentage', 0) > 10
    dynamic_good = dynamic_perf['final_error'] < 3.0
    
    overall_score = sum([accuracy_good, consistency_good, improvement_good, dynamic_good])
    
    if overall_score >= 3:
        assessment = "Excellent - Gaussian proxy significantly outperforms existing methods"
    elif overall_score >= 2:
        assessment = "Good - Gaussian proxy shows clear improvements"
    elif overall_score >= 1:
        assessment = "Fair - Some benefits but needs refinement"
    else:
        assessment = "Poor - Consider alternative approaches"
    
    summary['overall_assessment'] = {
        'score': overall_score,
        'max_score': 4,
        'assessment': assessment,
        'accuracy_good': accuracy_good,
        'consistency_good': consistency_good,
        'improvement_good': improvement_good,
        'dynamic_good': dynamic_good
    }
    
    # Print summary
    print(f"\n🎯 EXPERIMENT SUMMARY")
    print("=" * 25)
    print(f"Gaussian Proxy Performance:")
    print(f"  Mean Error: {gaussian_perf['mean_error_across_scenarios']:.2f} Hz")
    print(f"  Success Rate: {gaussian_perf['mean_success_rate']*100:.1f}%")
    print(f"  Consistency: {'Good' if consistency_good else 'Needs Improvement'}")
    
    if 'improvement_percentage' in comparison_perf:
        print(f"\nComparison vs Existing Methods:")
        print(f"  Improvement: {comparison_perf['improvement_percentage']:+.1f}%")
        print(f"  Better Performance: {'Yes' if comparison_perf['better_performance'] else 'No'}")
    
    print(f"\nDynamic Adaptation:")
    print(f"  Final Error: {dynamic_perf['final_error']:.2f} Hz")
    print(f"  Adaptation Time: {dynamic_perf['mean_adaptation_time']:.1f} iterations avg")
    
    print(f"\nOverall Assessment: {assessment}")
    
    return summary

def create_gaussian_results_plots(optimization_results, dynamic_results, best_proxy, 
                                trained_proxies, proxy_comparison):
    """Create comprehensive visualization of Gaussian proxy results"""
    
    # Create output directory
    output_dir = Path("output/gaussian_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Proxy comparison plot
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Gaussian Proxy Training and Comparison', fontsize=16)
    
    # Training quality comparison
    ax1 = axes[0, 0]
    proxy_names = list(trained_proxies.keys())
    r_squared_values = [trained_proxies[name]['proxy'].training_r_squared or 0 
                       for name in proxy_names]
    
    bars = ax1.bar(proxy_names, r_squared_values, alpha=0.7)
    ax1.set_ylabel('R-squared')
    ax1.set_title('(a) Training Quality')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # BEP prediction accuracy
    ax2 = axes[0, 1]
    bep_errors = [proxy_comparison['proxy_performance'][name]['bep_error'] 
                 for name in proxy_names]
    
    ax2.bar(proxy_names, bep_errors, alpha=0.7, color='orange')
    ax2.axhline(2.0, color='red', linestyle='--', label='Target (2Hz)')
    ax2.set_ylabel('BEP Error (Hz)')
    ax2.set_title('(b) BEP Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Correlation with true efficiency
    ax3 = axes[1, 0]
    correlations = [proxy_comparison['proxy_performance'][name]['correlation'] 
                   for name in proxy_names]
    
    ax3.bar(proxy_names, correlations, alpha=0.7, color='green')
    ax3.set_ylabel('Correlation with True Efficiency')
    ax3.set_title('(c) Proxy Quality')
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Best proxy training curve
    ax4 = axes[1, 1]
    if hasattr(best_proxy, 'training_flows') and best_proxy.training_flows:
        flows = best_proxy.training_flows
        proxies = best_proxy.training_proxies
        
        ax4.scatter(flows, proxies, alpha=0.6, label='Training Data')
        
        if best_proxy.is_trained:
            flow_smooth = np.linspace(min(flows), max(flows), 100)
            proxy_smooth = [best_proxy.eta_max * np.exp(-2 * best_proxy.sigma**2 * (q - best_proxy.Q_bep)**2) 
                           for q in flow_smooth]
            ax4.plot(flow_smooth, proxy_smooth, 'r-', linewidth=2, label='Gaussian Fit')
            ax4.axvline(best_proxy.Q_bep, color='green', linestyle='--', 
                       label=f'BEP: {best_proxy.Q_bep:.0f} m³/h')
    
    ax4.set_xlabel('Flow Rate (m³/h)')
    ax4.set_ylabel('Proxy Value')
    ax4.set_title('(d) Best Proxy Training')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'gaussian_proxy_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Optimization performance comparison
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Optimization Performance: Gaussian vs Existing Methods', fontsize=16)
    
    # Extract method performance data
    methods = ['gaussian_best', 'volumetric_original', 'original_q2p']
    method_labels = ['Gaussian', 'Volumetric', 'Original Q²/P']
    scenarios = list(optimization_results.keys())
    
    # Mean error comparison
    ax1 = axes[0, 0]
    method_errors = {method: [] for method in methods}
    
    for scenario in scenarios:
        for method in methods:
            if method in optimization_results[scenario]:
                method_errors[method].append(optimization_results[scenario][method]['mean_error'])
    
    x = np.arange(len(method_labels))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        scenario_errors = []
        for method in methods:
            if method in optimization_results[scenario]:
                scenario_errors.append(optimization_results[scenario][method]['mean_error'])
            else:
                scenario_errors.append(0)
        
        ax1.bar(x + i*width, scenario_errors, width, label=scenario, alpha=0.7)
    
    ax1.set_ylabel('Mean Error (Hz)')
    ax1.set_title('(a) Mean Error by Method and Scenario')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(method_labels)
    ax1.axhline(2.0, color='red', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Success rate comparison
    ax2 = axes[0, 1]
    for i, scenario in enumerate(scenarios):
        scenario_success = []
        for method in methods:
            if method in optimization_results[scenario]:
                scenario_success.append(optimization_results[scenario][method]['success_rate'] * 100)
            else:
                scenario_success.append(0)
        
        ax2.bar(x + i*width, scenario_success, width, label=scenario, alpha=0.7)
    
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('(b) Success Rate (±2Hz)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(method_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Dynamic adaptation results
    ax3 = axes[1, 0]
    if dynamic_results:
        iterations = [r['iteration'] for r in dynamic_results]
        errors = [r['error'] for r in dynamic_results]
        heads = [r['system_head'] for r in dynamic_results]
        
        # Plot error evolution
        ax3.plot(iterations, errors, 'b-', linewidth=2, label='BEP Error')
        ax3.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        
        # Mark head changes
        head_changes = [15, 25]
        for change_iter in head_changes:
            ax3.axvline(change_iter, color='orange', linestyle=':', linewidth=2)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('BEP Error (Hz)')
        ax3.set_title('(c) Dynamic Head Adaptation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # System head timeline
        ax4 = axes[1, 1]
        ax4.step(iterations, heads, where='post', linewidth=3, color='brown')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('System Head (m)')
        ax4.set_title('(d) System Head Changes')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'gaussian_optimization_results.png', dpi=300, bbox_inches='tight')
    
    print(f"Results plots saved to {output_dir}")
    
    plt.show()

# Import existing proxy functions for comparison
from src.proxy_functions import VolumetricEfficiencyProxy, OriginalProxy

if __name__ == "__main__":
    # Run the complete experiment
    results = run_gaussian_proxy_experiment()
    
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"Results saved and visualized.")
    print(f"Check 'output/gaussian_experiment/' for detailed plots and data.")