"""
Utility functions for Deep Hedging Experiment
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def print_performance_summary(metrics_dict, title="Performance Summary"):
    """Print formatted performance summary"""
    print(f"\n{title}")
    print("=" * 50)
    
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
    
    print("=" * 50)

def save_results_to_csv(results_dict, filename="hedging_results.csv"):
    """Save results to CSV file"""
    df = pd.DataFrame(results_dict).T
    df.to_csv(filename)
    print(f"Results saved to {filename}")

def load_results_from_csv(filename="hedging_results.csv"):
    """Load results from CSV file"""
    return pd.read_csv(filename, index_col=0)

def create_experiment_report(results_dict, config, filename="experiment_report.txt"):
    """Create a comprehensive experiment report"""
    with open(filename, 'w') as f:
        f.write("DEEP HEDGING EXPERIMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-" * 20 + "\n")
        f.write(f"Asset: {config.DATA_SYMBOL}\n")
        f.write(f"Derivative: European {config.OPTION_TYPE.upper()} Option\n")
        f.write(f"Risk Metric: CVaR {config.CVAR_CONFIDENCE*100}%\n")
        f.write(f"Training Episodes: {config.HEDGER_EPOCHS}\n")
        f.write(f"Sequence Length: {config.SEQUENCE_LENGTH}\n\n")
        
        f.write("HYPOTHESIS TESTING\n")
        f.write("-" * 20 + "\n")
        f.write("H0: No difference in CVaR between GAN-trained and GBM-trained hedgers\n")
        f.write("H1: Adversarial-trained hedger achieves 20% lower CVaR95\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for method, metrics in results_dict.items():
            f.write(f"\n{method}:\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
        
        f.write("\nCONCLUSION\n")
        f.write("-" * 20 + "\n")
        f.write("This report contains the experimental results for the deep hedging study.\n")
        f.write("For detailed analysis, refer to the generated plots and statistical tests.\n")
    
    print(f"Experiment report saved to {filename}") 