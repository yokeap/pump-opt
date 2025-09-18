# ==============================================================================
# FILE: src/experiments.py (Enhanced with comprehensive logging)
# ==============================================================================

import numpy as np
import pandas as pd
import time
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .pump_model import RealisticPumpSimulator, PumpMeasurement
from .optimizers import TPEOptimizer, ExtremumSeekingControl, RandomSearch, GridSearch
from .proxy_functions import VolumetricEfficiencyProxy, OriginalProxy, NormalizedProxy

@dataclass
class ExperimentResult:
    """Structure to hold experiment results with enhanced metadata"""
    experiment_name: str
    method_name: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None

class ExperimentLogger:
    """Comprehensive experiment logging system"""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_dir = Path(base_output_dir)
        self.setup_directories()
        
        # Create session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / "experiments" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment logging initialized: {self.session_dir}")
        
    def setup_directories(self):
        """Create directory structure for experiment logging"""
        dirs = [
            self.base_dir / "experiments",
            self.base_dir / "data" / "raw",
            self.base_dir / "data" / "processed", 
            self.base_dir / "results" / "json",
            self.base_dir / "results" / "csv",
            self.base_dir / "results" / "pickle",
            self.base_dir / "logs"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_result(self, result: ExperimentResult, save_formats=['json', 'pickle']):
        """Save experiment result in multiple formats"""
        
        # Create safe filename
        safe_name = f"{result.experiment_name}_{result.method_name}".replace(" ", "_").lower()
        timestamp = datetime.fromtimestamp(result.timestamp).strftime("%H%M%S")
        base_filename = f"{timestamp}_{safe_name}"
        
        saved_files = []
        
        try:
            # Save as JSON (human readable)
            if 'json' in save_formats:
                json_file = self.session_dir / f"{base_filename}.json"
                
                # Convert to JSON-serializable format
                json_data = {
                    'experiment_name': result.experiment_name,
                    'method_name': result.method_name,
                    'timestamp': result.timestamp,
                    'datetime': datetime.fromtimestamp(result.timestamp).isoformat(),
                    'duration_seconds': result.duration_seconds,
                    'success': result.success,
                    'error_message': result.error_message,
                    'metadata': result.metadata,
                    'results': self._make_json_serializable(result.results)
                }
                
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                saved_files.append(str(json_file))
            
            # Save as Pickle (preserves exact Python objects)
            if 'pickle' in save_formats:
                pickle_file = self.session_dir / f"{base_filename}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(result, f)
                saved_files.append(str(pickle_file))
            
            # Save detailed data as CSV if results contain arrays/lists
            if 'csv' in save_formats:
                csv_files = self._save_as_csv(result, base_filename)
                saved_files.extend(csv_files)
                
        except Exception as e:
            print(f"Warning: Failed to save experiment result: {e}")
            
        return saved_files
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, PumpMeasurement):
            return asdict(obj)
        else:
            return obj
    
    def _save_as_csv(self, result: ExperimentResult, base_filename: str):
        """Save tabular data as CSV files"""
        csv_files = []
        results = result.results
        
        try:
            # Save history data if available
            if 'history' in results and results['history']:
                history_df = pd.DataFrame(results['history'])
                csv_file = self.session_dir / f"{base_filename}_history.csv"
                history_df.to_csv(csv_file, index=False)
                csv_files.append(str(csv_file))
            
            # Save optimization data if available
            if 'optimization_data' in results and results['optimization_data']:
                opt_df = pd.DataFrame(results['optimization_data'])
                csv_file = self.session_dir / f"{base_filename}_optimization_data.csv"
                opt_df.to_csv(csv_file, index=False)
                csv_files.append(str(csv_file))
                
            # Save performance data if available
            if 'proxy_performance' in results and results['proxy_performance']:
                perf_df = pd.DataFrame(results['proxy_performance'])
                csv_file = self.session_dir / f"{base_filename}_proxy_performance.csv"
                perf_df.to_csv(csv_file, index=False)
                csv_files.append(str(csv_file))
                
            # Save method comparison data
            if 'method_performance' in results:
                all_methods_data = []
                for method, method_results in results['method_performance'].items():
                    for result_item in method_results:
                        result_item['method'] = method
                        all_methods_data.append(result_item)
                
                if all_methods_data:
                    methods_df = pd.DataFrame(all_methods_data)
                    csv_file = self.session_dir / f"{base_filename}_method_comparison.csv"
                    methods_df.to_csv(csv_file, index=False)
                    csv_files.append(str(csv_file))
                    
        except Exception as e:
            print(f"Warning: Failed to save some CSV files: {e}")
            
        return csv_files
    
    def save_session_summary(self, all_results: List[ExperimentResult]):
        """Save summary of all experiments in this session"""
        
        summary_data = {
            'session_id': self.session_id,
            'session_start': min(r.timestamp for r in all_results),
            'session_end': max(r.timestamp for r in all_results),
            'total_experiments': len(all_results),
            'successful_experiments': sum(1 for r in all_results if r.success),
            'total_duration_seconds': sum(r.duration_seconds for r in all_results),
            'experiments_summary': []
        }
        
        for result in all_results:
            summary_data['experiments_summary'].append({
                'experiment_name': result.experiment_name,
                'method_name': result.method_name,
                'success': result.success,
                'duration_seconds': result.duration_seconds,
                'timestamp': result.timestamp
            })
        
        # Save session summary
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
            
        print(f"Session summary saved: {summary_file}")
        return str(summary_file)

class ExperimentRunner:
    """Enhanced experiment runner with comprehensive logging"""
    
    def __init__(self, random_seed: int = 42, enable_logging: bool = True, output_dir: str = "output"):
        self.random_seed = random_seed
        self.enable_logging = enable_logging
        np.random.seed(random_seed)
        
        # Initialize logger
        if enable_logging:
            self.logger = ExperimentLogger(output_dir)
            print(f"Experiment logging enabled: {self.logger.session_dir}")
        else:
            self.logger = None
            
        self.experiment_results = []  # Store all results for session summary
        
    def _log_experiment(self, experiment_name: str, method_name: str, 
                       results: Dict[str, Any], metadata: Dict[str, Any],
                       start_time: float, success: bool = True, 
                       error_message: str = None):
        """Log experiment results"""
        
        if not self.enable_logging or not self.logger:
            return
            
        duration = time.time() - start_time
        
        experiment_result = ExperimentResult(
            experiment_name=experiment_name,
            method_name=method_name,
            results=results,
            metadata=metadata,
            timestamp=start_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message
        )
        
        # Save individual experiment
        saved_files = self.logger.save_experiment_result(experiment_result)
        
        # Store for session summary
        self.experiment_results.append(experiment_result)
        
        print(f"Experiment logged: {experiment_name} - {method_name}")
        print(f"  Saved files: {len(saved_files)}")
        print(f"  Duration: {duration:.1f} seconds")
        
        return experiment_result
        
    def validate_proxy_function(self,
                               proxy_class,
                               test_heads: List[float] = [20, 25, 30, 35, 40, 45, 50],
                               test_flows: List[float] = [0.7, 0.8, 0.9, 1.0, 1.1],
                               noise_levels: List[float] = [0.01, 0.02, 0.03, 0.05],
                               n_trials: int = 5) -> Dict[str, Any]:
        """
        Comprehensive validation of proxy function performance with logging
        """
        
        start_time = time.time()
        experiment_name = "proxy_validation"
        method_name = proxy_class().get_name()
        
        try:
            print("Running proxy function validation...")
            print(f"Testing {len(test_heads)} heads × {len(noise_levels)} noise levels × {n_trials} trials")
            
            results = {
                'proxy_performance': [],
                'correlation_analysis': [],
                'noise_sensitivity': [],
                'head_bias_analysis': [],
                'flow_sensitivity': []
            }
            
            # [Previous validation code remains the same...]
            # 1. Basic performance across different heads
            for head in test_heads:
                for trial in range(n_trials):
                    pump = RealisticPumpSimulator(system_head=head, noise_level=0.02)
                    optimizer = TPEOptimizer(proxy_function=proxy_class())
                    
                    trial_results = self._run_single_optimization(
                        pump, optimizer, max_iterations=15
                    )
                    
                    true_bep, true_eff = pump.get_true_bep()
                    best_freq, best_proxy = optimizer.get_best_bep()
                    error = abs(best_freq - true_bep) if best_freq else float('inf')
                    
                    results['proxy_performance'].append({
                        'head': head,
                        'trial': trial,
                        'true_bep': true_bep,
                        'predicted_bep': best_freq,
                        'error': error,
                        'true_efficiency': true_eff,
                        'best_achieved_efficiency': max([h['true_efficiency'] for h in trial_results]),
                        'iterations_to_convergence': self._find_convergence_iteration(trial_results),
                        'trial_history': trial_results  # Save complete history
                    })
            
            # 2. Correlation analysis
            for head in [25, 35, 45]:
                pump = RealisticPumpSimulator(system_head=head, noise_level=0.01)
                proxy_func = proxy_class()
                
                frequencies = np.linspace(25, 60, 50)
                proxy_values = []
                true_efficiencies = []
                
                for freq in frequencies:
                    measurement = pump.get_measurement(freq)
                    proxy_val = proxy_func.calculate(measurement)
                    proxy_values.append(proxy_val)
                    true_efficiencies.append(measurement.true_efficiency)
                
                correlation = np.corrcoef(proxy_values, true_efficiencies)[0, 1]
                
                results['correlation_analysis'].append({
                    'head': head,
                    'correlation': correlation,
                    'proxy_values': proxy_values,
                    'true_efficiencies': true_efficiencies,
                    'frequencies': frequencies.tolist()
                })
            
            # 3. Noise sensitivity analysis
            for noise_level in noise_levels:
                head_errors = []
                for head in [30, 40]:
                    pump = RealisticPumpSimulator(system_head=head, noise_level=noise_level)
                    optimizer = TPEOptimizer(proxy_function=proxy_class())
                    
                    trial_results = self._run_single_optimization(
                        pump, optimizer, max_iterations=12
                    )
                    
                    true_bep, _ = pump.get_true_bep()
                    best_freq, _ = optimizer.get_best_bep()
                    error = abs(best_freq - true_bep) if best_freq else float('inf')
                    head_errors.append(error)
                
                results['noise_sensitivity'].append({
                    'noise_level': noise_level,
                    'mean_error': np.mean(head_errors),
                    'std_error': np.std(head_errors),
                    'max_error': np.max(head_errors)
                })
            
            # 4. Head bias analysis
            low_heads = [20, 25, 30]
            high_heads = [40, 45, 50]
            low_head_errors = []
            high_head_errors = []
            
            for head_group, error_list in [(low_heads, low_head_errors), (high_heads, high_head_errors)]:
                for head in head_group:
                    for trial in range(3):
                        pump = RealisticPumpSimulator(system_head=head, noise_level=0.02)
                        optimizer = TPEOptimizer(proxy_function=proxy_class())
                        
                        self._run_single_optimization(pump, optimizer, max_iterations=12)
                        
                        true_bep, _ = pump.get_true_bep()
                        best_freq, _ = optimizer.get_best_bep()
                        error = abs(best_freq - true_bep) if best_freq else float('inf')
                        error_list.append(error)
            
            results['head_bias_analysis'] = {
                'low_head_mean_error': np.mean(low_head_errors),
                'high_head_mean_error': np.mean(high_head_errors),
                'bias': np.mean(high_head_errors) - np.mean(low_head_errors),
                'low_head_std': np.std(low_head_errors),
                'high_head_std': np.std(high_head_errors),
                'low_head_errors': low_head_errors,  # Raw data
                'high_head_errors': high_head_errors  # Raw data
            }
            
            # 5. Flow sensitivity
            for flow_factor in test_flows:
                rated_flow = 100 * flow_factor
                pump = RealisticPumpSimulator(system_head=35, rated_flow=rated_flow, noise_level=0.02)
                optimizer = TPEOptimizer(proxy_function=proxy_class(rated_flow))
                
                trial_results = self._run_single_optimization(pump, optimizer, max_iterations=12)
                
                true_bep, _ = pump.get_true_bep()
                best_freq, _ = optimizer.get_best_bep()
                error = abs(best_freq - true_bep) if best_freq else float('inf')
                
                results['flow_sensitivity'].append({
                    'flow_factor': flow_factor,
                    'rated_flow': rated_flow,
                    'error': error,
                    'trial_history': trial_results
                })
            
            # Calculate summary statistics
            proxy_performance = results['proxy_performance']
            results['summary'] = {
                'overall_mean_error': np.mean([r['error'] for r in proxy_performance]),
                'overall_std_error': np.std([r['error'] for r in proxy_performance]),
                'success_rate_2hz': sum(1 for r in proxy_performance if r['error'] < 2.0) / len(proxy_performance),
                'success_rate_5hz': sum(1 for r in proxy_performance if r['error'] < 5.0) / len(proxy_performance),
                'mean_correlation': np.mean([r['correlation'] for r in results['correlation_analysis']]),
                'proxy_name': proxy_class().get_name()
            }
            
            # Log experiment
            metadata = {
                'proxy_class_name': proxy_class.__name__,
                'test_heads': test_heads,
                'test_flows': test_flows, 
                'noise_levels': noise_levels,
                'n_trials': n_trials,
                'random_seed': self.random_seed
            }
            
            self._log_experiment(experiment_name, method_name, results, metadata, start_time, True)
            
            return results
            
        except Exception as e:
            print(f"Error in proxy validation: {e}")
            self._log_experiment(experiment_name, method_name, {}, {}, start_time, False, str(e))
            raise
    
    def dynamic_head_test(self,
                         initial_head: float = 30,
                         head_changes: List[Tuple[int, float]] = None,
                         optimization_method: str = 'TPE',
                         max_iterations: int = 80,
                         convergence_threshold: int = 35) -> Dict[str, Any]:
        """Test optimizer performance under REALISTIC dynamic head changes with logging"""
        
        start_time = time.time()
        experiment_name = "dynamic_head_test"
        method_name = f"{optimization_method}_realistic"
        
        try:
            # Realistic head change schedule
            if head_changes is None:
                head_changes = [(50, 35), (70, 40)]
            
            print(f"Running REALISTIC dynamic head test...")
            print(f"TPE Convergence time: ~{convergence_threshold} iterations")
            print(f"Head changes: {head_changes} (slow, realistic schedule)")
            
            # Setup pump and optimizer
            pump = RealisticPumpSimulator(system_head=initial_head, noise_level=0.025)
            
            if optimization_method == 'TPE':
                optimizer = TPEOptimizer(
                    proxy_function=VolumetricEfficiencyProxy(),
                    max_freq_change=1.5
                )
            elif optimization_method == 'ESC':
                optimizer = ExtremumSeekingControl(
                    proxy_function=VolumetricEfficiencyProxy(),
                    step_size=1.5
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # Track results with detailed logging
            results = {
                'history': [],
                'head_changes': [],
                'adaptation_performance': [],
                'true_bep_tracking': [],
                'convergence_analysis': [],
                'realistic_timeline': {
                    'scenario': 'Gradual head changes over days',
                    'time_scale': '10-minute intervals',
                    'total_duration_hours': max_iterations * 10 / 60,
                    'application': 'Long-term efficiency optimization'
                }
            }
            
            head_change_schedule = dict(head_changes)
            
            print(f"Starting optimization with initial head: {initial_head}m")
            
            for iteration in range(1, max_iterations + 1):
                # Check for head change
                if iteration in head_change_schedule:
                    new_head = head_change_schedule[iteration]
                    old_head = pump.current_head
                    old_bep, _ = pump.get_true_bep()
                    
                    pump.set_system_head(new_head)
                    new_bep, _ = pump.get_true_bep()
                    
                    results['head_changes'].append({
                        'iteration': iteration,
                        'real_time_hours': iteration * 10 / 60,
                        'old_head': old_head,
                        'new_head': new_head,
                        'old_bep': old_bep,
                        'new_bep': new_bep,
                        'bep_shift': new_bep - old_bep,
                        'change_type': 'gradual_system_change'
                    })
                
                # Run optimization step
                suggested_freq = optimizer.suggest_frequency()
                measurement = pump.get_measurement(suggested_freq)
                optimizer.update(suggested_freq, measurement)
                
                # Get current best estimate
                best_bep_estimate = optimizer.get_best_bep()
                true_bep, true_eff = pump.get_true_bep()
                
                # Store detailed results
                iteration_result = {
                    'iteration': iteration,
                    'real_time_hours': iteration * 10 / 60,
                    'frequency': suggested_freq,
                    'proxy_value': optimizer.proxy_function.calculate(measurement),
                    'true_efficiency': measurement.true_efficiency,
                    'system_head': pump.current_head,
                    'true_bep': true_bep,
                    'predicted_bep': best_bep_estimate[0] if best_bep_estimate else None,
                    'bep_error': abs(best_bep_estimate[0] - true_bep) if best_bep_estimate else float('inf'),
                    'measurement_data': asdict(measurement)  # Complete measurement data
                }
                
                results['history'].append(iteration_result)
                results['true_bep_tracking'].append({
                    'iteration': iteration,
                    'true_bep': true_bep,
                    'system_head': pump.current_head
                })
                
                if iteration % 10 == 0:
                    print(f"  Hour {iteration * 10 / 60:4.1f} (iter {iteration:2d}): "
                          f"freq={suggested_freq:.1f}Hz, error={iteration_result['bep_error']:.1f}Hz")
            
            # Enhanced adaptation analysis
            for change_info in results['head_changes']:
                change_iteration = change_info['iteration']
                adaptation_window = min(convergence_threshold, max_iterations - change_iteration)
                post_change_history = [h for h in results['history'] 
                                     if change_iteration < h['iteration'] <= change_iteration + adaptation_window]
                
                if post_change_history:
                    adaptation_errors = [h['bep_error'] for h in post_change_history]
                    final_error = post_change_history[-1]['bep_error']
                    
                    # Find stable adaptation time
                    stable_iterations = []
                    for i, h in enumerate(post_change_history):
                        if h['bep_error'] < 2.0:
                            next_errors = [post_change_history[j]['bep_error'] 
                                         for j in range(i+1, min(i+4, len(post_change_history)))]
                            if all(e < 3.0 for e in next_errors):
                                adaptation_iteration = h['iteration'] - change_iteration
                                stable_iterations.append(adaptation_iteration)
                                break
                    
                    results['adaptation_performance'].append({
                        'change_iteration': change_iteration,
                        'change_time_hours': change_iteration * 10 / 60,
                        'head_change': f"{change_info['old_head']}m → {change_info['new_head']}m",
                        'bep_shift': change_info['bep_shift'],
                        'mean_adaptation_error': np.mean(adaptation_errors),
                        'final_error': final_error,
                        'adaptation_iterations': stable_iterations[0] if stable_iterations else None,
                        'adaptation_hours': stable_iterations[0] * 10 / 60 if stable_iterations else None,
                        'adaptation_success': final_error < 2.0,
                        'adaptation_window': adaptation_window,
                        'adaptation_history': post_change_history  # Detailed adaptation data
                    })
            
            # Calculate summary
            all_errors = [h['bep_error'] for h in results['history'] if h['bep_error'] != float('inf')]
            
            results['summary'] = {
                'method': optimization_method,
                'initial_head': initial_head,
                'total_iterations': max_iterations,
                'total_hours': max_iterations * 10 / 60,
                'num_head_changes': len(head_changes),
                'overall_mean_error': np.mean(all_errors) if all_errors else float('inf'),
                'overall_std_error': np.std(all_errors) if all_errors else 0,
                'final_error': all_errors[-1] if all_errors else float('inf'),
                'adaptation_success_rate': sum(1 for a in results['adaptation_performance'] 
                                             if a['adaptation_success']) / len(results['adaptation_performance']) if results['adaptation_performance'] else 0,
                'mean_adaptation_hours': np.mean([a['adaptation_hours'] for a in results['adaptation_performance'] 
                                                if a['adaptation_hours'] is not None]) if results['adaptation_performance'] else None
            }
            
            # Log experiment with detailed metadata
            metadata = {
                'initial_head': initial_head,
                'head_changes': head_changes,
                'optimization_method': optimization_method,
                'max_iterations': max_iterations,
                'convergence_threshold': convergence_threshold,
                'update_interval_minutes': 10,
                'total_duration_hours': max_iterations * 10 / 60,
                'random_seed': self.random_seed,
                'pump_noise_level': 0.025
            }
            
            self._log_experiment(experiment_name, method_name, results, metadata, start_time, True)
            
            return results
            
        except Exception as e:
            print(f"Error in dynamic head test: {e}")
            self._log_experiment(experiment_name, method_name, {}, {}, start_time, False, str(e))
            raise
    
    def realtime_demo(self,
                     duration_hours: float = 8.0,
                     update_interval_minutes: float = 10.0,
                     scenario_type: str = 'maintenance_optimization',
                     disturbances: List[Tuple] = None) -> Dict[str, Any]:
        """REALISTIC long-term optimization demo with comprehensive logging"""
        
        start_time = time.time()
        experiment_name = "longterm_optimization"
        method_name = f"TPE_{scenario_type}"
        
        try:
            # Realistic scenarios
            if scenario_type == 'maintenance_optimization':
                if disturbances is None:
                    disturbances = [
                        (2*3600, 'head_drift', 32),
                        (5*3600, 'measurement_degradation', 0.03),
                        (6.5*3600, 'head_drift', 28)
                    ]
                scenario_description = "Long-term optimization during normal operations"
                
            elif scenario_type == 'seasonal_adaptation':
                if disturbances is None:
                    disturbances = [
                        (3*3600, 'head_change', 35),
                        (7*3600, 'head_change', 40)
                    ]
                scenario_description = "Adaptation to gradually changing water levels"
            else:
                if disturbances is None:
                    disturbances = []
                scenario_description = "Custom optimization scenario"
            
            print(f"Running REALISTIC long-term optimization demo...")
            print(f"Scenario: {scenario_description}")
            print(f"Duration: {duration_hours} hours")
            
            # Setup
            pump = RealisticPumpSimulator(system_head=30, noise_level=0.02)
            optimizer = TPEOptimizer(
                proxy_function=VolumetricEfficiencyProxy(), 
                max_freq_change=1.0
            )
            
            results = {
                'timeline': [],
                'disturbances': [],
                'performance_metrics': [],
                'optimization_data': [],  # Changed from real_time_data
                'scenario_info': {
                    'type': scenario_type,
                    'description': scenario_description,
                    'time_scale': 'hours',
                    'application_context': 'Long-term efficiency optimization',
                    'tpe_suitability': 'high'
                }
            }
            
            duration_seconds = duration_hours * 3600
            update_interval_seconds = update_interval_minutes * 60
            iteration = 0
            
            # Create disturbance schedule
            disturbance_schedule = {}
            for dist_time, dist_type, dist_value in disturbances:
                disturbance_schedule[dist_time] = (dist_type, dist_value)
            
            print("Starting long-term optimization...")
            
            # Simulate the optimization process
            simulated_time = 0
            while simulated_time < duration_seconds:
                iteration += 1
                current_hours = simulated_time / 3600
                
                # Check for disturbances
                for scheduled_time, (dist_type, dist_value) in disturbance_schedule.items():
                    if abs(simulated_time - scheduled_time) < update_interval_seconds / 2:
                        if dist_type == 'head_change' or dist_type == 'head_drift':
                            old_head = pump.current_head
                            pump.set_system_head(dist_value)
                            results['disturbances'].append({
                                'time_hours': current_hours,
                                'time_seconds': simulated_time,
                                'type': dist_type,
                                'old_value': old_head,
                                'new_value': dist_value
                            })
                            
                        elif dist_type == 'measurement_degradation':
                            old_noise = pump.noise_level
                            pump.noise_level = dist_value
                            results['disturbances'].append({
                                'time_hours': current_hours,
                                'time_seconds': simulated_time,
                                'type': dist_type,
                                'old_value': old_noise,
                                'new_value': dist_value
                            })
                        
                        # Remove from schedule
                        del disturbance_schedule[scheduled_time]
                        break
                
                # Optimization step
                suggested_freq = optimizer.suggest_frequency()
                measurement = pump.get_measurement(suggested_freq)
                optimizer.update(suggested_freq, measurement)
                
                # Get current status
                best_estimate = optimizer.get_best_bep()
                true_bep, _ = pump.get_true_bep()
                
                # Store comprehensive optimization data
                data_point = {
                    'time_hours': current_hours,
                    'time_seconds': simulated_time,
                    'iteration': iteration,
                    'frequency': suggested_freq,
                    'flow': measurement.flow,
                    'power': measurement.power,
                    'efficiency': measurement.true_efficiency,
                    'system_head': pump.current_head,
                    'true_bep': true_bep,
                    'predicted_bep': best_estimate[0] if best_estimate else None,
                    'error': abs(best_estimate[0] - true_bep) if best_estimate else float('inf'),
                    'proxy_value': optimizer.proxy_function.calculate(measurement),
                    'convergence_indicator': iteration >= 30,
                    'measurement_data': asdict(measurement)  # Complete measurement record
                }
                
                results['optimization_data'].append(data_point)
                
                # Progress update
                if iteration % max(1, int(60 / update_interval_minutes)) == 0 or iteration <= 3:
                    error = data_point['error']
                    convergence_status = "Converged" if iteration >= 30 and error < 2.0 else "Converging"
                    print(f"  Hour {current_hours:4.1f}: freq={suggested_freq:.1f}Hz, "
                          f"error={error:.1f}Hz [{convergence_status}]")
                
                # Advance time
                simulated_time += update_interval_seconds
            
            # Calculate comprehensive performance metrics
            final_data = results['optimization_data']
            final_errors = [d['error'] for d in final_data if d['error'] != float('inf')]
            converged_data = [d for d in final_data if d['iteration'] >= 30]
            converged_errors = [d['error'] for d in converged_data if d['error'] != float('inf')]
            
            results['performance_metrics'] = {
                'scenario_type': scenario_type,
                'total_duration_hours': duration_hours,
                'total_iterations': len(final_data),
                'update_frequency_per_hour': 60 / update_interval_minutes,
                'tpe_convergence_time_hours': 30 * update_interval_minutes / 60,
                'final_error': final_errors[-1] if final_errors else float('inf'),
                'mean_error_overall': np.mean(final_errors) if final_errors else float('inf'),
                'mean_error_converged': np.mean(converged_errors) if converged_errors else float('inf'),
                'error_improvement': final_errors[0] - final_errors[-1] if len(final_errors) > 1 else 0,
                'converged_stability': np.std(converged_errors) if len(converged_errors) > 5 else float('inf'),
                'disturbances_handled': len(results['disturbances']),
                'avg_adaptation_time_hours': self._calculate_adaptation_times_realistic(results),
                'performance_assessment': {
                    'overall': 'Good' if (converged_errors and np.mean(converged_errors) < 2.0) else 'Needs Improvement',
                    'convergence': 'Achieved' if (converged_errors and np.mean(converged_errors) < 3.0) else 'Partial',
                    'stability': 'High' if (converged_errors and np.std(converged_errors) < 1.0) else 'Medium',
                    'suitability': 'High for long-term optimization, Low for rapid response'
                }
            }
            
            # Log experiment with comprehensive metadata
            metadata = {
                'scenario_type': scenario_type,
                'scenario_description': scenario_description,
                'duration_hours': duration_hours,
                'update_interval_minutes': update_interval_minutes,
                'total_optimization_steps': len(final_data),
                'disturbances_planned': len(disturbances),
                'disturbances_executed': len(results['disturbances']),
                'convergence_threshold_iterations': 30,
                'random_seed': self.random_seed,
                'pump_initial_head': 30,
                'pump_noise_level': 0.02
            }
            
            self._log_experiment(experiment_name, method_name, results, metadata, start_time, True)
            
            return results
            
        except Exception as e:
            print(f"Error in long-term demo: {e}")
            self._log_experiment(experiment_name, method_name, {}, {}, start_time, False, str(e))
            raise
    
    def comprehensive_comparison(self,
                               methods: List[str] = ['TPE', 'ESC', 'Random', 'GridSearch'],
                               test_scenarios: List[Dict] = None,
                               iterations: int = 25,
                               n_trials: int = 5) -> Dict[str, Any]:
        """Comprehensive comparison of optimization algorithms with detailed logging"""
        
        start_time = time.time()
        experiment_name = "comprehensive_comparison"
        method_name = "multi_algorithm"
        
        try:
            if test_scenarios is None:
                test_scenarios = [
                    {'head': 30, 'noise': 0.02, 'name': 'Baseline'},
                    {'head': 45, 'noise': 0.02, 'name': 'High Head'},
                    {'head': 25, 'noise': 0.05, 'name': 'Noisy'},
                    {'head': 35, 'noise': 0.02, 'name': 'Medium Head'}
                ]
            
            print(f"Running comprehensive comparison...")
            print(f"Methods: {methods}")
            print(f"Scenarios: {len(test_scenarios)} × {n_trials} trials × {iterations} iterations")
            
            results = {
                'method_performance': {},
                'scenario_analysis': {},
                'convergence_analysis': {},
                'robustness_analysis': {},
                'detailed_trial_data': []  # Store all individual trial data
            }
            
            # Initialize results structure
            for method in methods:
                results['method_performance'][method] = []
            
            # Run experiments with detailed logging
            total_experiments = len(methods) * len(test_scenarios) * n_trials
            completed = 0
            
            for method in methods:
                method_results = []
                
                for scenario in test_scenarios:
                    scenario_results = []
                    
                    for trial in range(n_trials):
                        completed += 1
                        trial_start_time = time.time()
                        
                        print(f"  Progress: {completed}/{total_experiments} - {method} on {scenario['name']} (trial {trial+1})")
                        
                        try:
                            # Setup pump
                            pump = RealisticPumpSimulator(
                                system_head=scenario['head'],
                                noise_level=scenario['noise']
                            )
                            
                            # Setup optimizer
                            proxy = VolumetricEfficiencyProxy()
                            if method == 'TPE':
                                optimizer = TPEOptimizer(proxy_function=proxy)
                            elif method == 'ESC':
                                optimizer = ExtremumSeekingControl(proxy_function=proxy)
                            elif method == 'Random':
                                optimizer = RandomSearch(proxy_function=proxy)
                            elif method == 'GridSearch':
                                optimizer = GridSearch(proxy_function=proxy)
                            else:
                                raise ValueError(f"Unknown method: {method}")
                            
                            # Run optimization with detailed tracking
                            trial_history = self._run_single_optimization(
                                pump, optimizer, max_iterations=iterations
                            )
                            
                            # Analyze results
                            true_bep, true_eff = pump.get_true_bep()
                            best_freq, best_proxy = optimizer.get_best_bep()
                            
                            final_error = abs(best_freq - true_bep) if best_freq else float('inf')
                            best_achieved_eff = max([h['true_efficiency'] for h in trial_history])
                            convergence_iter = self._find_convergence_iteration(trial_history, target_error=2.0)
                            
                            trial_result = {
                                'method': method,
                                'scenario': scenario['name'],
                                'head': scenario['head'],
                                'noise': scenario['noise'],
                                'trial': trial,
                                'true_bep': true_bep,
                                'predicted_bep': best_freq,
                                'final_error': final_error,
                                'true_efficiency': true_eff,
                                'best_achieved_efficiency': best_achieved_eff,
                                'convergence_iteration': convergence_iter,
                                'history': trial_history,
                                'trial_duration_seconds': time.time() - trial_start_time,
                                'pump_parameters': {
                                    'system_head': scenario['head'],
                                    'noise_level': scenario['noise'],
                                    'true_bep_frequency': true_bep,
                                    'max_efficiency': true_eff
                                }
                            }
                            
                            scenario_results.append(trial_result)
                            method_results.append(trial_result)
                            
                            # Store detailed trial data
                            results['detailed_trial_data'].append(trial_result)
                            
                        except Exception as trial_error:
                            print(f"    Error in trial: {trial_error}")
                            # Store failed trial info
                            failed_trial = {
                                'method': method,
                                'scenario': scenario['name'],
                                'trial': trial,
                                'error': str(trial_error),
                                'success': False
                            }
                            results['detailed_trial_data'].append(failed_trial)
                    
                    # Scenario-specific analysis
                    successful_results = [r for r in scenario_results if 'final_error' in r]
                    if successful_results:
                        scenario_errors = [r['final_error'] for r in successful_results]
                        scenario_convs = [r['convergence_iteration'] for r in successful_results 
                                        if r['convergence_iteration'] is not None]
                        
                        if scenario['name'] not in results['scenario_analysis']:
                            results['scenario_analysis'][scenario['name']] = {}
                        
                        results['scenario_analysis'][scenario['name']][method] = {
                            'mean_error': np.mean(scenario_errors),
                            'std_error': np.std(scenario_errors),
                            'success_rate': sum(1 for e in scenario_errors if e < 2.0) / len(scenario_errors),
                            'mean_convergence': np.mean(scenario_convs) if scenario_convs else None,
                            'convergence_rate': len(scenario_convs) / len(successful_results),
                            'trial_count': len(successful_results)
                        }
                
                results['method_performance'][method] = method_results
            
            # Cross-method analysis
            for method in methods:
                method_data = [r for r in results['method_performance'][method] if 'final_error' in r]
                if method_data:
                    errors = [r['final_error'] for r in method_data]
                    convergences = [r['convergence_iteration'] for r in method_data 
                                  if r['convergence_iteration'] is not None]
                    
                    results['convergence_analysis'][method] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'median_error': np.median(errors),
                        'success_rate_2hz': sum(1 for e in errors if e < 2.0) / len(errors),
                        'success_rate_5hz': sum(1 for e in errors if e < 5.0) / len(errors),
                        'mean_convergence_time': np.mean(convergences) if convergences else None,
                        'convergence_rate': len(convergences) / len(method_data),
                        'total_trials': len(method_data)
                    }
                    
                    # Robustness analysis
                    scenario_means = []
                    for scenario in test_scenarios:
                        scenario_errors = [r['final_error'] for r in method_data 
                                         if r['scenario'] == scenario['name']]
                        if scenario_errors:
                            scenario_means.append(np.mean(scenario_errors))
                    
                    if scenario_means:
                        results['robustness_analysis'][method] = {
                            'cross_scenario_std': np.std(scenario_means),
                            'worst_scenario_error': np.max(scenario_means),
                            'best_scenario_error': np.min(scenario_means)
                        }
            
            # Log comprehensive comparison
            metadata = {
                'methods_compared': methods,
                'test_scenarios': test_scenarios,
                'iterations_per_trial': iterations,
                'trials_per_scenario': n_trials,
                'total_experiments': total_experiments,
                'successful_experiments': len([t for t in results['detailed_trial_data'] 
                                             if t.get('success', True)]),
                'random_seed': self.random_seed
            }
            
            self._log_experiment(experiment_name, method_name, results, metadata, start_time, True)
            
            return results
            
        except Exception as e:
            print(f"Error in comprehensive comparison: {e}")
            self._log_experiment(experiment_name, method_name, {}, {}, start_time, False, str(e))
            raise
    
    def save_session_summary(self):
        """Save summary of all experiments in this session"""
        if self.enable_logging and self.logger and self.experiment_results:
            summary_file = self.logger.save_session_summary(self.experiment_results)
            print(f"Session summary with {len(self.experiment_results)} experiments saved to: {summary_file}")
            return summary_file
        return None
    
    def _calculate_adaptation_times_realistic(self, results: Dict) -> float:
        """Calculate realistic adaptation times (hours, not minutes)"""
        if not results['disturbances']:
            return 0.0
        
        adaptation_times = []
        
        for disturbance in results['disturbances']:
            dist_time_hours = disturbance['time_hours']
            
            # Find data points after disturbance
            post_dist_data = [d for d in results['optimization_data'] 
                            if d['time_hours'] > dist_time_hours]
            
            if not post_dist_data:
                continue
            
            # Find when error becomes acceptable AND stays stable
            for i, data_point in enumerate(post_dist_data):
                if data_point['error'] < 2.0 and i >= 10:
                    # Check next few points for stability
                    next_points = post_dist_data[i:i+5]
                    if all(p['error'] < 3.0 for p in next_points):
                        adaptation_time_hours = data_point['time_hours'] - dist_time_hours
                        adaptation_times.append(adaptation_time_hours)
                        break
        
        return np.mean(adaptation_times) if adaptation_times else float('inf')
    
    def _run_single_optimization(self, pump, optimizer, max_iterations: int) -> List[Dict]:
        """Run a single optimization trial with detailed logging"""
        history = []
        
        for iteration in range(1, max_iterations + 1):
            frequency = optimizer.suggest_frequency()
            measurement = pump.get_measurement(frequency)
            optimizer.update(frequency, measurement)
            
            history.append({
                'iteration': iteration,
                'frequency': frequency,
                'proxy_value': optimizer.proxy_function.calculate(measurement),
                'true_efficiency': measurement.true_efficiency,
                'measurement': asdict(measurement)  # Complete measurement data
            })
        
        return history
    
    def _find_convergence_iteration(self, history: List[Dict], target_error: float = 2.0) -> Optional[int]:
        """Find iteration when optimization converged"""
        if len(history) < 5:
            return None
        
        proxy_values = [h['proxy_value'] for h in history]
        
        # Find when proxy value stabilizes
        for i in range(5, len(proxy_values)):
            recent_values = proxy_values[i-5:i]
            if np.std(recent_values) < 0.1 * np.mean(recent_values):
                return i
        
        return None