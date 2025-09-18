# ==============================================================================
# FILE: src/utils.py
# ==============================================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from datetime import datetime
import logging
from pathlib import Path
import platform
import sys

def setup_publication_style():
    """Setup matplotlib style for publication-quality figures"""
    
    # Set publication-ready style
    plt.style.use('default')
    
    # Font settings
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
    })
    
    # Figure settings
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.figsize': [8, 6],
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
    })
    
    # Axes settings
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Line and marker settings
    plt.rcParams.update({
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'patch.linewidth': 1.0,
    })
    
    # Legend and tick settings
    plt.rcParams.update({
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'legend.framealpha': 0.9,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
    
    # Set color palette for consistency
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A2B8', '#FD7E14']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    print("Publication style configured for high-quality figures")

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """Setup logging for experiment tracking"""
    
    logger = logging.getLogger('BEP_Research')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_output_directories(base_dir: str = 'output') -> Dict[str, Path]:
    """Create directory structure for output files"""
    
    base_path = Path(base_dir)
    
    directories = {
        'base': base_path,
        'figures': base_path / 'figures',
        'data': base_path / 'data',
        'reports': base_path / 'reports',
        'logs': base_path / 'logs'
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories created in: {base_path.absolute()}")
    return directories

def save_experiment_data(data: Dict[str, Any], 
                        filename: str,
                        output_dir: Path = None,
                        include_metadata: bool = True) -> Path:
    """Save experiment data with metadata"""
    
    if output_dir is None:
        output_dir = Path('output/data')
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata if requested
    if include_metadata:
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
        }
        data['_metadata'] = metadata
    
    # Save as JSON
    filepath = output_dir / f"{filename}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Experiment data saved: {filepath}")
    return filepath

def load_experiment_data(filepath: Path) -> Dict[str, Any]:
    """Load experiment data from file"""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Remove metadata from main data
    metadata = data.pop('_metadata', None)
    if metadata:
        print(f"Loaded data from: {metadata.get('timestamp', 'unknown time')}")
    
    return data

def calculate_statistical_metrics(errors: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistical metrics for errors"""
    
    errors = np.array(errors)
    errors = errors[~np.isnan(errors)]  # Remove NaN values
    
    if len(errors) == 0:
        return {'count': 0}
    
    metrics = {
        'count': len(errors),
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'q25': np.percentile(errors, 25),
        'q75': np.percentile(errors, 75),
        'iqr': np.percentile(errors, 75) - np.percentile(errors, 25),
        'success_rate_1hz': np.sum(errors < 1.0) / len(errors),
        'success_rate_2hz': np.sum(errors < 2.0) / len(errors),
        'success_rate_5hz': np.sum(errors < 5.0) / len(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors))
    }
    
    return metrics

def generate_comparison_table(results: Dict[str, List[float]], 
                            metric_name: str = "Error (Hz)") -> pd.DataFrame:
    """Generate comparison table for different methods"""
    
    table_data = []
    
    for method_name, values in results.items():
        metrics = calculate_statistical_metrics(values)
        
        table_data.append({
            'Method': method_name,
            f'Mean {metric_name}': f"{metrics.get('mean', 0):.2f}",
            f'Std {metric_name}': f"{metrics.get('std', 0):.2f}",
            f'Median {metric_name}': f"{metrics.get('median', 0):.2f}",
            'Success Rate (≤2Hz)': f"{metrics.get('success_rate_2hz', 0)*100:.1f}%",
            'Success Rate (≤5Hz)': f"{metrics.get('success_rate_5hz', 0)*100:.1f}%",
            'Min': f"{metrics.get('min', 0):.2f}",
            'Max': f"{metrics.get('max', 0):.2f}",
            'Count': metrics.get('count', 0)
        })
    
    return pd.DataFrame(table_data)

def export_latex_table(df: pd.DataFrame, 
                      filename: str,
                      caption: str = "",
                      label: str = "",
                      output_dir: Path = None) -> Path:
    """Export DataFrame as LaTeX table for publication"""
    
    if output_dir is None:
        output_dir = Path('output/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure LaTeX output
    latex_str = df.to_latex(
        index=False,
        float_format="%.2f",
        column_format='l' + 'c' * (len(df.columns) - 1),
        escape=False
    )
    
    # Add caption and label if provided
    if caption:
        latex_str = latex_str.replace('\\end{tabular}', 
            f'\\end{{tabular}}\n\\caption{{{caption}}}')
    
    if label:
        latex_str = latex_str.replace('\\end{table}', 
            f'\\label{{{label}}}\n\\end{{table}}')
    
    # Save to file
    filepath = output_dir / f"{filename}.tex"
    with open(filepath, 'w') as f:
        f.write(latex_str)
    
    print(f"LaTeX table exported: {filepath}")
    return filepath

def calculate_convergence_metrics(history: List[Dict], 
                                true_bep: float, 
                                tolerance: float = 2.0) -> Dict[str, Any]:
    """Calculate convergence metrics from optimization history"""
    
    frequencies = [h['frequency'] for h in history]
    errors = [abs(f - true_bep) for f in frequencies]
    
    # Find convergence iteration (first time within tolerance)
    convergence_iteration = None
    for i, error in enumerate(errors):
        if error <= tolerance:
            convergence_iteration = i + 1
            break
    
    # Calculate cumulative best (showing improvement over time)
    cumulative_best = []
    best_so_far = float('inf')
    for error in errors:
        best_so_far = min(best_so_far, error)
        cumulative_best.append(best_so_far)
    
    return {
        'convergence_iteration': convergence_iteration,
        'final_error': errors[-1],
        'best_error': min(errors),
        'convergence_rate': len([e for e in errors if e <= tolerance]) / len(errors),
        'cumulative_best': cumulative_best,
        'improvement_rate': (errors[0] - errors[-1]) / len(errors) if len(errors) > 1 else 0
    }

def robust_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """Calculate robust correlation metrics"""
    
    x = np.array(x)
    y = np.array(y)
    
    # Remove NaN and infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return {'pearson': 0.0, 'spearman': 0.0, 'count': len(x)}
    
    # Pearson correlation
    try:
        pearson_corr = np.corrcoef(x, y)[0, 1]
    except:
        pearson_corr = 0.0
    
    # Spearman correlation (rank-based, more robust)
    try:
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(x, y)
    except:
        spearman_corr = 0.0
    
    return {
        'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'count': len(x)
    }

def detect_outliers(data: List[float], method: str = 'iqr') -> Tuple[List[int], Dict[str, float]]:
    """Detect outliers in data using specified method"""
    
    data = np.array(data)
    outlier_indices = []
    stats = {}
    
    if method == 'iqr':
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        
        stats = {
            'method': 'iqr',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'n_outliers': len(outlier_indices)
        }
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        threshold = 3.0
        
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        
        stats = {
            'method': 'zscore',
            'threshold': threshold,
            'mean': np.mean(data),
            'std': np.std(data),
            'n_outliers': len(outlier_indices)
        }
    
    return outlier_indices, stats

def format_scientific_notation(value: float, precision: int = 2) -> str:
    """Format number in scientific notation for publication"""
    
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def create_method_summary(results: Dict[str, Any]) -> str:
    """Create a formatted summary of experimental methods"""
    
    summary_lines = [
        "EXPERIMENTAL SETUP SUMMARY",
        "=" * 40,
        "",
        f"Pump Model: {results.get('pump_type', 'Realistic Centrifugal')}",
        f"System Heads: {results.get('test_heads', [25, 30, 35, 40, 45])} m",
        f"Noise Levels: {results.get('noise_levels', [0.01, 0.02, 0.05])}",
        f"Optimization Methods: {results.get('methods', ['TPE', 'ESC', 'Random'])}",
        f"Iterations per Trial: {results.get('max_iterations', 25)}",
        f"Number of Trials: {results.get('n_trials', 5)}",
        f"Proxy Function: {results.get('proxy_name', 'Volumetric Efficiency')}",
        "",
        "PERFORMANCE METRICS",
        "-" * 20,
        f"Target Accuracy: ±2 Hz (Success)",
        f"Acceptable Accuracy: ±5 Hz (Good)",
        f"Convergence Criterion: Proxy stabilization",
        "",
        f"Total Experiments: {results.get('total_experiments', 'N/A')}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ]
    
    return "\n".join(summary_lines)

def validate_experimental_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate experimental data for completeness and consistency"""
    
    issues = {
        'warnings': [],
        'errors': [],
        'info': []
    }
    
    # Check for required fields
    required_fields = ['method_performance', 'summary']
    for field in required_fields:
        if field not in data:
            issues['errors'].append(f"Missing required field: {field}")
    
    # Check data consistency
    if 'method_performance' in data:
        for method, results in data['method_performance'].items():
            if not results:
                issues['warnings'].append(f"No results found for method: {method}")
            
            # Check for outliers in errors
            errors = [r.get('final_error', float('inf')) for r in results if isinstance(r, dict)]
            if errors:
                outlier_indices, outlier_stats = detect_outliers(errors)
                if outlier_stats['n_outliers'] > len(errors) * 0.1:  # More than 10% outliers
                    issues['warnings'].append(f"High outlier rate in {method}: {outlier_stats['n_outliers']}/{len(errors)} ({outlier_stats['n_outliers']/len(errors)*100:.1f}%)")
    
    # Check for data completeness
    if 'summary' in data:
        summary = data['summary']
        if summary.get('overall_mean_error', 0) > 10:
            issues['warnings'].append("High overall mean error (>10 Hz)")
        
        if summary.get('success_rate_2hz', 0) < 0.5:
            issues['warnings'].append("Low success rate (<50%)")
    
    # Information about data quality
    issues['info'].append(f"Data validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return issues

class ProgressTracker:
    """Track and display experiment progress"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = None, message: str = ""):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current_step > 0:
            eta = elapsed * (self.total_steps - self.current_step) / self.current_step
        else:
            eta = "Unknown"
        
        print(f"\r{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps}) | ETA: {eta} | {message}", end="")
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
            
    def finish(self, final_message: str = "Complete"):
        """Mark as finished"""
        self.current_step = self.total_steps
        elapsed = datetime.now() - self.start_time
        print(f"\r{self.description}: {final_message} | Total time: {elapsed}")

def benchmark_performance():
    """Benchmark system performance for optimization"""
    
    print("Running system performance benchmark...")
    
    # Test numerical operations
    start_time = time.time()
    test_array = np.random.randn(10000, 1000)
    np.linalg.svd(test_array[:100, :100])  # Matrix decomposition
    numerical_time = time.time() - start_time
    
    # Test optimization operations
    start_time = time.time()
    from scipy.optimize import minimize_scalar
    for _ in range(100):
        minimize_scalar(lambda x: (x-2)**2, bounds=(0, 4), method='bounded')
    optimization_time = time.time() - start_time
    
    results = {
        'numerical_operations_time': numerical_time,
        'optimization_time': optimization_time,
        'estimated_experiment_time': optimization_time * 50,  # Rough estimate
        'system_ready': numerical_time < 5.0 and optimization_time < 2.0
    }
    
    print(f"Numerical operations: {numerical_time:.2f}s")
    print(f"Optimization operations: {optimization_time:.2f}s")
    print(f"System ready: {'Yes' if results['system_ready'] else 'No'}")
    
    return results