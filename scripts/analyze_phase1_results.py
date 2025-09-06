#!/usr/bin/env python3
"""
Phase 1.5 Results Analysis Script

This script analyzes the results from Phase 1.5 baseline experiments and extracts
key metrics for comparison and decision gate evaluation.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Any


def extract_metrics_from_log(stdout: str) -> Dict[str, float]:
    """Extract key metrics from experiment log output"""
    metrics = {}
    
    # Extract CVaR
    cvar_match = re.search(r'final_hedger_cvar: ([\d.]+)', stdout)
    if cvar_match:
        metrics['cvar_95'] = float(cvar_match.group(1))
    
    # Extract final reward
    reward_match = re.search(r'final_hedger_reward: ([-\d.]+)', stdout)
    if reward_match:
        metrics['final_reward'] = float(reward_match.group(1))
    
    # Extract GAN losses
    gan_d_match = re.search(r'final_gan_d_loss: ([\d.]+)', stdout)
    if gan_d_match:
        metrics['gan_d_loss'] = float(gan_d_match.group(1))
    
    gan_g_match = re.search(r'final_gan_g_loss: ([\d.]+)', stdout)
    if gan_g_match:
        metrics['gan_g_loss'] = float(gan_g_match.group(1))
    
    return metrics


def analyze_matrix_results(matrix_dir: str) -> pd.DataFrame:
    """Analyze results from a matrix execution directory"""
    matrix_path = Path(matrix_dir)
    summary_file = matrix_path / "matrix_summary.yaml"
    
    if not summary_file.exists():
        print(f"Matrix summary not found: {summary_file}")
        return pd.DataFrame()
    
    with open(summary_file, 'r') as f:
        summary = yaml.safe_load(f)
    
    results = []
    
    for experiment in summary['experiments']:
        if experiment['status'] != 'completed':
            continue
            
        # Parse run_id to extract parameters
        run_id = experiment['run_id']
        parts = run_id.split('_')
        
        # Extract model, TC, frequency, reward
        model = parts[0] + '_' + parts[1] if len(parts) > 1 else parts[0]  # Handle c_static, rl_gbm, etc.
        tc_match = re.search(r'tc(\d+)bps', run_id)
        tc_bps = int(tc_match.group(1)) if tc_match else 0
        
        freq_match = re.search(r'freq(\d+)min', run_id)
        freq_min = int(freq_match.group(1)) if freq_match else 8
        
        reward = parts[-1] if parts[-1] in ['cvar95', 'mean_variance'] else 'cvar95'
        
        # Extract metrics from log
        metrics = extract_metrics_from_log(experiment['stdout'])
        
        # Create result row
        result = {
            'run_id': run_id,
            'model': model,
            'tc_bps': tc_bps,
            'freq_min': freq_min,
            'reward': reward,
            'status': experiment['status'],
            'duration_seconds': (pd.to_datetime(experiment['end_time']) - 
                               pd.to_datetime(experiment['start_time'])).total_seconds(),
            **metrics
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def evaluate_decision_gates(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Phase 1 decision gates"""
    gates = {}
    
    # Gate G-1: RL viability - RL-GBM must beat C-static on CVaR 95 by ≥ 15% at baseline TC
    c_static_baseline = df[(df['model'] == 'c_static') & (df['tc_bps'] == 2) & (df['freq_min'] == 8)]
    rl_gbm_baseline = df[(df['model'] == 'rl_gbm') & (df['tc_bps'] == 2) & (df['freq_min'] == 8)]
    
    if len(c_static_baseline) > 0 and len(rl_gbm_baseline) > 0:
        c_static_cvar = c_static_baseline['cvar_95'].iloc[0]
        rl_gbm_cvar = rl_gbm_baseline['cvar_95'].iloc[0]
        improvement = (c_static_cvar - rl_gbm_cvar) / c_static_cvar * 100
        
        gates['G-1_RL_viability'] = {
            'passed': improvement >= 15,
            'improvement_pct': improvement,
            'c_static_cvar': c_static_cvar,
            'rl_gbm_cvar': rl_gbm_cvar,
            'threshold': 15
        }
    
    # Gate G-2: GAN value-add - RL-GAN must beat RL-GBM on CVaR 95 by ≥ 10% with no Sharpe penalty
    rl_gan_baseline = df[(df['model'] == 'rl_gan') & (df['tc_bps'] == 2) & (df['freq_min'] == 8)]
    
    if len(rl_gan_baseline) > 0:
        rl_gan_cvar = rl_gan_baseline['cvar_95'].iloc[0]
        gan_improvement = (rl_gbm_cvar - rl_gan_cvar) / rl_gbm_cvar * 100
        
        gates['G-2_GAN_value_add'] = {
            'passed': gan_improvement >= 10,
            'improvement_pct': gan_improvement,
            'rl_gbm_cvar': rl_gbm_cvar,
            'rl_gan_cvar': rl_gan_cvar,
            'threshold': 10
        }
    
    # Gate G-3: Robustness - Gains persist in ≥ 80% of ablative settings
    # This would require running the full matrix to evaluate
    
    return gates


def print_analysis_summary(df: pd.DataFrame, gates: Dict[str, Any]):
    """Print a summary of the analysis"""
    print("🔬 Phase 1.5 Baseline Analysis Summary")
    print("=" * 50)
    
    print(f"\n📊 Experiment Results:")
    print(f"   Total experiments: {len(df)}")
    print(f"   Models tested: {df['model'].unique()}")
    print(f"   Transaction cost levels: {sorted(df['tc_bps'].unique())} bps")
    print(f"   Frequencies tested: {sorted(df['freq_min'].unique())} min")
    
    print(f"\n📈 Key Metrics by Model (Baseline: 2bps TC, 8min freq):")
    baseline = df[(df['tc_bps'] == 2) & (df['freq_min'] == 8)]
    
    for _, row in baseline.iterrows():
        print(f"   {row['model'].upper()}: CVaR@95 = {row['cvar_95']:.2f}, "
              f"Final Reward = {row['final_reward']:.2f}")
    
    print(f"\n🎯 Decision Gate Evaluation:")
    for gate_name, gate_result in gates.items():
        status = "✅ PASSED" if gate_result['passed'] else "❌ FAILED"
        print(f"   {gate_name}: {status}")
        if 'improvement_pct' in gate_result:
            print(f"     Improvement: {gate_result['improvement_pct']:.1f}% "
                  f"(threshold: {gate_result['threshold']}%)")
    
    print(f"\n💡 Recommendations:")
    if all(gate['passed'] for gate in gates.values()):
        print("   🎉 All Phase 1 decision gates passed! Ready for Phase 2.")
    else:
        print("   ⚠️  Some decision gates failed. Review architecture before Phase 2.")
    
    # Show parameter sensitivity
    print(f"\n🔍 Parameter Sensitivity Analysis:")
    for param in ['tc_bps', 'freq_min']:
        if param in df.columns:
            param_values = sorted(df[param].unique())
            print(f"   {param}: {param_values}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase 1.5 baseline results")
    parser.add_argument("matrix_dir", help="Directory containing matrix results")
    
    args = parser.parse_args()
    
    # Analyze results
    df = analyze_matrix_results(args.matrix_dir)
    
    if df.empty:
        print("No results to analyze")
        return
    
    # Evaluate decision gates
    gates = evaluate_decision_gates(df)
    
    # Print summary
    print_analysis_summary(df, gates)
    
    # Save detailed results
    output_file = Path(args.matrix_dir) / "analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📁 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
