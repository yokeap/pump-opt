# ==============================================================================
# Data-Driven Pump BEP Tracking Without Head Measurement - Clean Version
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from src.pump_model import RealisticPumpSimulator
from src.optimizers import TPEOptimizer, ExtremumSeekingControl
from src.proxy_functions import OriginalProxy

class BEPTrackingExperiments:
    """Clean experiments for BEP tracking research paper"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir.absolute()}")
    
    def run_static_tests(self, test_heads: List[float] = [8, 12, 16, 20], 
                        n_trials: int = 5, max_iterations: int = 25) -> Dict[str, Any]:
        """Static BEP tracking tests - Low-head pump configuration"""
        
        print("\n" + "="*60)
        print("STATIC BEP TRACKING TESTS - LOW-HEAD PUMP")
        print("="*60)
        
        all_results = []
        method_summaries = {'TPE': [], 'ESC': []}
        
        for head in test_heads:
            print(f"\nTesting System Head: {head}m")
            print("-" * 30)
            
            for method_name in ['TPE', 'ESC']:
                method_errors = []
                method_iterations = []
                method_final_efficiencies = []
                
                print(f"  {method_name} Method:")
                
                for trial in range(n_trials):
                    # Setup low-head pump
                    pump = RealisticPumpSimulator(system_head=head, noise_level=0.01)
                    true_bep, true_eff = pump.get_true_bep()
                    
                    # Setup optimizer
                    proxy = OriginalProxy()
                    if method_name == 'TPE':
                        optimizer = TPEOptimizer(proxy_function=proxy)
                    else:
                        optimizer = ExtremumSeekingControl(proxy_function=proxy)
                    
                    # Run optimization
                    for iteration in range(1, max_iterations + 1):
                        frequency = optimizer.suggest_frequency()
                        measurement = pump.get_measurement(frequency)
                        optimizer.update(frequency, measurement)
                    
                    # Get results
                    best_freq, best_proxy = optimizer.get_best_bep()
                    error = abs(best_freq - true_bep) if best_freq else float('inf')
                    
                    # Find convergence iteration
                    convergence_iter = self._find_convergence_iteration(optimizer.history, true_bep)
                    
                    # Get final efficiency achieved
                    final_efficiency = max([h['true_efficiency'] for h in optimizer.history])
                    
                    trial_result = {
                        'head': head,
                        'method': method_name,
                        'trial': trial + 1,
                        'true_bep': true_bep,
                        'predicted_bep': best_freq,
                        'error': error,
                        'convergence_iteration': convergence_iter,
                        'final_efficiency': final_efficiency,
                        'total_iterations': max_iterations
                    }
                    
                    all_results.append(trial_result)
                    method_errors.append(error)
                    method_iterations.append(convergence_iter if convergence_iter else max_iterations)
                    method_final_efficiencies.append(final_efficiency)
                    
                    print(f"    Trial {trial+1}: Error={error:.2f}Hz, Conv@iter={convergence_iter}, Eff={final_efficiency:.3f}")
                
                # Method summary
                method_summary = {
                    'head': head,
                    'method': method_name,
                    'mean_error': np.mean(method_errors),
                    'std_error': np.std(method_errors),
                    'success_rate_2hz': sum(1 for e in method_errors if e <= 2.0) / len(method_errors),
                    'mean_convergence_iter': np.mean(method_iterations),
                    'mean_final_efficiency': np.mean(method_final_efficiencies),
                    'n_trials': n_trials
                }
                
                method_summaries[method_name].append(method_summary)
                
                print(f"    Summary: Error={method_summary['mean_error']:.2f}±{method_summary['std_error']:.2f}Hz, "
                      f"Success={method_summary['success_rate_2hz']*100:.0f}%")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "csv" / "static_test_detailed.csv", index=False)
        
        all_summaries = []
        for method, summaries in method_summaries.items():
            all_summaries.extend(summaries)
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(self.output_dir / "csv" / "static_test_summary.csv", index=False)
        
        # Generate plots
        self._plot_static_results(results_df, summary_df)
        
        # Print summary
        self._print_static_summary(method_summaries)
        
        return {
            'detailed_results': all_results,
            'method_summaries': method_summaries,
            'test_parameters': {
                'test_heads': test_heads,
                'n_trials': n_trials,
                'max_iterations': max_iterations
            }
        }
    
    def run_dynamic_tests(self, initial_head: float = 10, 
                         head_changes: List[Tuple[int, float]] = [(40, 15), (65, 20)],
                         max_iterations: int = 90) -> Dict[str, Any]:
        """Dynamic BEP tracking tests - Low-head pump configuration"""
        
        print("\n" + "="*60)
        print("DYNAMIC BEP TRACKING TESTS - LOW-HEAD PUMP")
        print("="*60)
        print(f"Initial head: {initial_head}m")
        print(f"Head changes: {head_changes} (iteration, new_head)")
        print(f"Total duration: ~{max_iterations * 10 / 60:.1f} hours")
        
        all_results = []
        method_summaries = []
        
        for method_name in ['TPE', 'ESC']:
            print(f"\n{method_name} Dynamic Test:")
            print("-" * 20)
            
            # Setup low-head pump
            pump = RealisticPumpSimulator(system_head=initial_head, noise_level=0.01)
            
            proxy = OriginalProxy()
            if method_name == 'TPE':
                optimizer = TPEOptimizer(proxy_function=proxy, max_freq_change=1.0)
            else:
                optimizer = ExtremumSeekingControl(proxy_function=proxy, step_size=1.0)
            
            # Track results
            results = {
                'history': [],
                'head_changes': [],
                'adaptation_performance': []
            }
            
            head_change_schedule = dict(head_changes)
            
            for iteration in range(1, max_iterations + 1):
                # Check for head change
                if iteration in head_change_schedule:
                    new_head = head_change_schedule[iteration]
                    old_head = pump.current_head
                    old_bep, _ = pump.get_true_bep()
                    
                    pump.set_system_head(new_head)
                    new_bep, _ = pump.get_true_bep()
                    
                    change_info = {
                        'iteration': iteration,
                        'real_time_hours': iteration * 10 / 60,
                        'old_head': old_head,
                        'new_head': new_head,
                        'old_bep': old_bep,
                        'new_bep': new_bep,
                        'bep_shift': new_bep - old_bep
                    }
                    results['head_changes'].append(change_info)
                    print(f"  Iteration {iteration}: Head {old_head}m -> {new_head}m, BEP {old_bep:.1f}Hz -> {new_bep:.1f}Hz")
                
                # Optimization step
                suggested_freq = optimizer.suggest_frequency()
                measurement = pump.get_measurement(suggested_freq)
                optimizer.update(suggested_freq, measurement)
                
                # Get current best
                best_bep_estimate = optimizer.get_best_bep()
                true_bep, true_eff = pump.get_true_bep()
                
                # Store results
                iteration_result = {
                    'method': method_name,
                    'iteration': iteration,
                    'real_time_hours': iteration * 10 / 60,
                    'frequency': suggested_freq,
                    'system_head': pump.current_head,
                    'true_bep': true_bep,
                    'predicted_bep': best_bep_estimate[0] if best_bep_estimate else None,
                    'bep_error': abs(best_bep_estimate[0] - true_bep) if best_bep_estimate else float('inf'),
                    'efficiency': measurement.true_efficiency
                }
                
                results['history'].append(iteration_result)
                all_results.append(iteration_result)
                
                if iteration % 10 == 0:
                    print(f"    Hour {iteration * 10 / 60:4.1f}: freq={suggested_freq:.1f}Hz, error={iteration_result['bep_error']:.1f}Hz")
            
            # Adaptation analysis
            for change_info in results['head_changes']:
                change_iteration = change_info['iteration']
                adaptation_window = min(20, max_iterations - change_iteration)
                post_change_history = [h for h in results['history'] 
                                     if change_iteration < h['iteration'] <= change_iteration + adaptation_window]
                
                if post_change_history:
                    adaptation_errors = [h['bep_error'] for h in post_change_history]
                    final_error = post_change_history[-1]['bep_error']
                    
                    # Find adaptation time
                    adaptation_iteration = None
                    for i, h in enumerate(post_change_history):
                        if h['bep_error'] < 3.0:
                            next_errors = [post_change_history[j]['bep_error'] 
                                         for j in range(i+1, min(i+3, len(post_change_history)))]
                            if len(next_errors) == 0 or np.mean(next_errors) < 4.0:
                                adaptation_iteration = i + 1
                                break
                    
                    adaptation_perf = {
                        'change_iteration': change_iteration,
                        'head_change': f"{change_info['old_head']}m -> {change_info['new_head']}m",
                        'adaptation_iterations': adaptation_iteration,
                        'adaptation_hours': adaptation_iteration * 10 / 60 if adaptation_iteration else None,
                        'adaptation_success': final_error < 3.0,
                        'method': method_name
                    }
                    
                    results['adaptation_performance'].append(adaptation_perf)
                    
                    if adaptation_iteration:
                        print(f"    Adapted in {adaptation_iteration} iterations ({adaptation_iteration * 10 / 60:.1f} hours)")
                    else:
                        print(f"    Failed to adapt within {adaptation_window} iterations")
            
            # Method summary
            all_errors = [h['bep_error'] for h in results['history'] if h['bep_error'] != float('inf')]
            
            method_summary = {
                'method': method_name,
                'overall_mean_error': np.mean(all_errors) if all_errors else float('inf'),
                'final_error': all_errors[-1] if all_errors else float('inf'),
                'adaptation_success_rate': sum(1 for a in results['adaptation_performance'] 
                                             if a['adaptation_success']) / len(results['adaptation_performance']) if results['adaptation_performance'] else 0,
                'mean_adaptation_hours': np.mean([a['adaptation_hours'] for a in results['adaptation_performance'] 
                                                if a['adaptation_hours'] is not None]) if results['adaptation_performance'] else None
            }
            
            method_summaries.append(method_summary)
            print(f"  Summary: Mean={method_summary['overall_mean_error']:.2f}Hz, Final={method_summary['final_error']:.2f}Hz")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "csv" / "dynamic_test_detailed.csv", index=False)
        
        # Generate plots
        self._plot_dynamic_results(results_df)
        
        # Print summary
        self._print_dynamic_summary(results_df, method_summaries)
        
        return {
            'detailed_results': all_results,
            'method_summaries': method_summaries,
            'test_parameters': {
                'initial_head': initial_head,
                'head_changes': head_changes,
                'max_iterations': max_iterations
            }
        }
    
    def _find_convergence_iteration(self, history: List[Dict], true_bep: float, tolerance: float = 2.0) -> int:
        """Find iteration when optimizer converged"""
        for i in range(3, len(history)):
            recent_errors = []
            for j in range(i - 2, i + 1):
                freq = history[j]['frequency']
                error = abs(freq - true_bep)
                recent_errors.append(error)
            
            if all(e <= tolerance for e in recent_errors):
                return i + 1
        
        return None
    
    def _plot_static_results(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Generate static test plots"""
        
        # Plot 1: BEP Error by Head
        plt.figure(figsize=(10, 6))
        
        methods = ['TPE', 'ESC']
        heads = sorted(results_df['head'].unique())
        
        for i, method in enumerate(methods):
            method_data = results_df[results_df['method'] == method]
            head_means = []
            head_stds = []
            
            for head in heads:
                head_errors = method_data[method_data['head'] == head]['error']
                head_means.append(head_errors.mean())
                head_stds.append(head_errors.std())
            
            x_pos = np.arange(len(heads)) + i * 0.35
            plt.bar(x_pos, head_means, 0.35, yerr=head_stds, label=method, alpha=0.8, capsize=5)
        
        plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Success (2Hz)')
        plt.xlabel('System Head (m)')
        plt.ylabel('BEP Prediction Error (Hz)')
        plt.title('Static BEP Tracking Performance - Low-Head Pump')
        plt.xticks(np.arange(len(heads)) + 0.175, heads)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "static_bep_error.png", dpi=300)
        plt.close()
        
        # Plot 2: Success Rate
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            method_summary = summary_df[summary_df['method'] == method]
            x_pos = np.arange(len(heads)) + i * 0.35
            success_rates = method_summary['success_rate_2hz'].values * 100
            plt.bar(x_pos, success_rates, 0.35, label=method, alpha=0.8)
        
        plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
        plt.xlabel('System Head (m)')
        plt.ylabel('Success Rate (%)')
        plt.title('BEP Tracking Success Rate - Low-Head Pump')
        plt.xticks(np.arange(len(heads)) + 0.175, heads)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "static_success_rate.png", dpi=300)
        plt.close()
    
    def _plot_dynamic_results(self, results_df: pd.DataFrame):
        """Generate dynamic test plots"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        methods = results_df['method'].unique()
        colors = ['blue', 'red']
        
        for method, color in zip(methods, colors):
            method_data = results_df[results_df['method'] == method]
            
            ax1.plot(method_data['real_time_hours'], method_data['predicted_bep'], 
                    'o-', color=color, label=f'{method} Prediction', alpha=0.8, markersize=3)
            ax1.plot(method_data['real_time_hours'], method_data['true_bep'], 
                    '--', color='black', label='True BEP', linewidth=2, alpha=0.7)
            
            ax2.plot(method_data['real_time_hours'], method_data['bep_error'], 
                    'o-', color=color, label=f'{method} Error', alpha=0.8, markersize=3)
        
        # Mark head changes
        head_change_times = []
        prev_head = None
        for idx, row in results_df.iterrows():
            current_head = row['system_head']
            if prev_head is not None and current_head != prev_head:
                head_change_times.append(row['real_time_hours'])
            prev_head = current_head
        
        head_change_times = sorted(list(set(head_change_times)))
        for t in head_change_times:
            ax1.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
            ax2.axvline(x=t, color='red', linestyle=':', alpha=0.7, linewidth=2)
        
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Dynamic BEP Tracking - Low-Head Pump')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Success (2Hz)')
        ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Acceptable (5Hz)')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('BEP Error (Hz)')
        ax2.set_title('Tracking Error Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "dynamic_timeline.png", dpi=300)
        plt.close()
    
    def _print_static_summary(self, method_summaries: Dict[str, List[Dict]]):
        """Print static test summary"""
        
        print("\n" + "="*60)
        print("STATIC TEST SUMMARY - LOW-HEAD PUMP")
        print("="*60)
        
        methods = ['TPE', 'ESC']
        heads = [8, 12, 16, 20]
        
        print(f"\nOverall Performance:")
        print("-" * 40)
        print(f"{'Method':<8} {'Mean Error (Hz)':<15} {'Success Rate':<12}")
        
        for method in methods:
            summaries = method_summaries[method]
            all_errors = [s['mean_error'] for s in summaries]
            all_success = [s['success_rate_2hz'] for s in summaries]
            
            mean_error = np.mean(all_errors)
            mean_success = np.mean(all_success) * 100
            
            print(f"{method:<8} {mean_error:<15.2f} {mean_success:<12.0f}%")
        
        print(f"\nDetailed Results by Head:")
        print("-" * 60)
        print(f"{'Head':<6} {'Method':<8} {'Error':<12} {'Success':<10} {'Efficiency':<12}")
        
        for head in heads:
            for method in methods:
                summary = next(s for s in method_summaries[method] if s['head'] == head)
                print(f"{head:<6} {method:<8} {summary['mean_error']:<12.2f} "
                      f"{summary['success_rate_2hz']*100:<10.0f}% {summary['mean_final_efficiency']:<12.3f}")
    
    def _print_dynamic_summary(self, results_df: pd.DataFrame, method_summaries: List[Dict]):
        """Print dynamic test summary"""
        
        print("\n" + "="*60)
        print("DYNAMIC TEST SUMMARY - LOW-HEAD PUMP")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print("-" * 30)
        print(f"{'Method':<8} {'Mean Error':<12} {'Final Error':<12} {'Adaptation':<12}")
        
        for summary in method_summaries:
            method = summary['method']
            mean_error = summary['overall_mean_error']
            final_error = summary['final_error']
            adapt_success = summary['adaptation_success_rate']
            
            print(f"{method:<8} {mean_error:<12.2f} {final_error:<12.2f} {adapt_success*100:<12.0f}%")
        
        # Compare methods
        tpe_summary = next(s for s in method_summaries if s['method'] == 'TPE')
        esc_summary = next(s for s in method_summaries if s['method'] == 'ESC')
        
        print(f"\nCOMPARISON:")
        print("-" * 12)
        better_method = 'TPE' if tpe_summary['overall_mean_error'] < esc_summary['overall_mean_error'] else 'ESC'
        print(f"Better performance: {better_method}")
        
        final_tpe = tpe_summary['final_error']
        final_esc = esc_summary['final_error']
        best_final = min(final_tpe, final_esc)
        
        if best_final <= 2.0:
            print("CONCLUSION: Excellent - Ready for deployment")
        elif best_final <= 5.0:
            print("CONCLUSION: Good - Suitable with monitoring")
        else:
            print("CONCLUSION: Needs improvement")


def main():
    """Main function to run BEP tracking experiments"""
    
    print("Data-Driven Pump BEP Tracking Without Head Measurement")
    print("="*58)
    print("Research Experiments - Low-Head Submersible Pump")
    
    # Initialize experiment runner
    experiments = BEPTrackingExperiments(output_dir="results")
    
    # Run static tests with low-head pump parameters
    static_results = experiments.run_static_tests(
        test_heads=[8, 12, 16, 20],  # Low-head range
        n_trials=7,
        max_iterations=35
    )
    
    # Run dynamic tests with low-head pump parameters
    dynamic_results = experiments.run_dynamic_tests(
        initial_head=10,  # Start at 10m
        head_changes=[(30, 20), (60, 30)],  # 10->15->20m progression
        max_iterations=90
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETION")
    print("="*60)
    print("Files saved in results/ directory:")
    print("• CSV files: results/csv/")
    print("• Plots: results/plots/")
    print("\nReady for paper analysis!")


if __name__ == "__main__":
    main()