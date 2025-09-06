#!/usr/bin/env python3
"""
Phase 1.5 Baseline Experimental Matrix Runner

This script launches the complete experimental matrix for Phase 1.5:
- 4 model variants (C-static, C-dyn, RL-GBM, RL-GAN)
- 3 transaction cost levels (0 bps, 2 bps, 10 bps)
- 3 hedge frequencies (1 min, 8 min, 30 min)
- 1 reward function (CVaR95 baseline)
- 1 ablation reward function (mean-variance)

Total: 48 runs (4 × 3 × 3 × 1 + 4 × 3 × 3 × 1 = 48)

Usage:
    python scripts/run_matrix.py
    python scripts/run_matrix.py --dry-run  # Show commands without executing
    python scripts/run_matrix.py --model rl_gan  # Run only RL-GAN variants
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import yaml
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from deephedge.utils.seed import set_seed


class Phase1MatrixRunner:
    """Manages the Phase 1.5 experimental matrix execution"""
    
    def __init__(self, dry_run: bool = False, output_dir: str = "runs/phase1_matrix"):
        self.dry_run = dry_run
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Define the experimental matrix
        self.model_variants = ["c_static", "c_dyn", "rl_gbm", "rl_gan"]
        self.tc_levels = [0, 2, 10]  # bps
        self.frequencies = [1, 8, 30]  # minutes
        self.reward_functions = ["cvar95", "mean_variance"]
        
        # Track run results
        self.run_log = []
        self.start_time = datetime.now()
        
    def generate_experiment_matrix(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations"""
        experiments = []
        
        for model in self.model_variants:
            for tc_bps in self.tc_levels:
                for freq_min in self.frequencies:
                    for reward in self.reward_functions:
                        # Skip certain combinations that don't make sense
                        if model in ["c_static", "c_dyn"] and reward == "mean_variance":
                            continue  # Classical models don't use reward functions
                        
                        experiment = {
                            "model": model,
                            "tc_bps": tc_bps,
                            "freq_min": freq_min,
                            "reward": reward,
                            "run_id": f"{model}_tc{tc_bps}bps_freq{freq_min}min_{reward}",
                            "description": f"{model.upper()} with {tc_bps}bps TC, {freq_min}min frequency, {reward} reward"
                        }
                        experiments.append(experiment)
        
        return experiments
    
    def create_hydra_overrides(self, experiment: Dict[str, Any]) -> List[str]:
        """Create Hydra command line overrides for an experiment"""
        overrides = []
        
        # Model variant
        overrides.append(f"model={experiment['model']}")
        
        # Transaction costs
        overrides.append(f"environment.transaction_cost={experiment['tc_bps']/10000}")
        
        # Hedge frequency (resample frequency)
        overrides.append(f"data.resample_freq={experiment['freq_min']}T")
        
        # Reward function (for RL models)
        if experiment['model'].startswith('rl_'):
            if experiment['reward'] == 'cvar95':
                overrides.append("training.hedger.cvar_alpha=0.95")
            elif experiment['reward'] == 'mean_variance':
                overrides.append("training.hedger.reward_type=mean_variance")
                overrides.append("training.hedger.risk_aversion=1.0")
        
        # Seed for reproducibility
        overrides.append(f"experiment.seed={42}")
        
        # Output directory
        overrides.append(f"hydra.run.dir={self.output_dir}/{experiment['run_id']}")
        
        return overrides
    
    def run_experiment(self, experiment: Dict[str, Any]) -> bool:
        """Run a single experiment"""
        run_id = experiment['run_id']
        print(f"\n🚀 Running experiment: {run_id}")
        print(f"   Description: {experiment['description']}")
        
        # Create Hydra overrides
        overrides = self.create_hydra_overrides(experiment)
        override_str = " ".join([f'"{o}"' for o in overrides])
        
        # Build command
        cmd = f"python -m src.deephedge.train {override_str}"
        
        if self.dry_run:
            print(f"   Command: {cmd}")
            return True
        
        # Create experiment log entry
        run_entry = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "command": cmd,
            "overrides": overrides,
            "status": "running"
        }
        self.run_log.append(run_entry)
        
        try:
            # Run the experiment
            print(f"   Executing: {cmd}")
            result = subprocess.run(
                cmd, 
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Update run log
            run_entry["end_time"] = datetime.now().isoformat()
            run_entry["exit_code"] = result.returncode
            run_entry["stdout"] = result.stdout
            run_entry["stderr"] = result.stderr
            
            if result.returncode == 0:
                run_entry["status"] = "completed"
                print(f"   ✅ Experiment {run_id} completed successfully")
                return True
            else:
                run_entry["status"] = "failed"
                print(f"   ❌ Experiment {run_id} failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            run_entry["status"] = "timeout"
            run_entry["end_time"] = datetime.now().isoformat()
            print(f"   ⏰ Experiment {run_id} timed out after 1 hour")
            return False
        except Exception as e:
            run_entry["status"] = "error"
            run_entry["end_time"] = datetime.now().isoformat()
            run_entry["error"] = str(e)
            print(f"   💥 Experiment {run_id} encountered error: {e}")
            return False
    
    def run_matrix(self, model_filter: str = None) -> None:
        """Run the complete experimental matrix"""
        print("🔬 Phase 1.5 Baseline Experimental Matrix")
        print("=" * 50)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        
        # Generate experiment matrix
        experiments = self.generate_experiment_matrix()
        
        # Apply model filter if specified
        if model_filter:
            experiments = [e for e in experiments if e['model'] == model_filter]
            print(f"Filtered to {model_filter} variants only")
        
        print(f"Total experiments to run: {len(experiments)}")
        
        # Run experiments
        successful_runs = 0
        failed_runs = 0
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n📊 Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
            
            if self.run_experiment(experiment):
                successful_runs += 1
            else:
                failed_runs += 1
        
        # Generate summary report
        self.generate_summary_report(successful_runs, failed_runs)
        
        print(f"\n🎯 Matrix execution completed!")
        print(f"   Successful: {successful_runs}")
        print(f"   Failed: {failed_runs}")
        print(f"   Total time: {datetime.now() - self.start_time}")
    
    def generate_summary_report(self, successful_runs: int, failed_runs: int) -> None:
        """Generate a summary report of the matrix execution"""
        report_path = self.output_dir / "matrix_summary.yaml"
        
        summary = {
            "phase": "1.5_baseline_matrix",
            "execution_time": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds()
            },
            "results": {
                "total_experiments": len(self.run_log),
                "successful": successful_runs,
                "failed": failed_runs,
                "success_rate": successful_runs / len(self.run_log) if self.run_log else 0
            },
            "experiments": self.run_log,
            "matrix_config": {
                "model_variants": self.model_variants,
                "tc_levels": self.tc_levels,
                "frequencies": self.frequencies,
                "reward_functions": self.reward_functions
            }
        }
        
        with open(report_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        print(f"📋 Summary report saved to: {report_path}")
    
    def show_matrix(self) -> None:
        """Display the experimental matrix without running"""
        print("🔬 Phase 1.5 Experimental Matrix Overview")
        print("=" * 50)
        
        experiments = self.generate_experiment_matrix()
        
        print(f"Total experiments: {len(experiments)}")
        print("\nMatrix breakdown:")
        
        for model in self.model_variants:
            model_experiments = [e for e in experiments if e['model'] == model]
            print(f"\n{model.upper()}:")
            
            for tc_bps in self.tc_levels:
                for freq_min in self.frequencies:
                    tc_freq_experiments = [
                        e for e in model_experiments 
                        if e['tc_bps'] == tc_bps and e['freq_min'] == freq_min
                    ]
                    
                    if tc_freq_experiments:
                        rewards = [e['reward'] for e in tc_freq_experiments]
                        print(f"  TC: {tc_bps}bps, Freq: {freq_min}min → Rewards: {rewards}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 Baseline Experimental Matrix Runner")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--model", choices=["c_static", "c_dyn", "rl_gbm", "rl_gan"], 
                       help="Run only specific model variant")
    parser.add_argument("--output-dir", default="runs/phase1_matrix", 
                       help="Output directory for results")
    parser.add_argument("--show-matrix", action="store_true", 
                       help="Display matrix overview without running")
    
    args = parser.parse_args()
    
    # Create runner
    runner = Phase1MatrixRunner(
        dry_run=args.dry_run,
        output_dir=args.output_dir
    )
    
    if args.show_matrix:
        runner.show_matrix()
    else:
        runner.run_matrix(model_filter=args.model)


if __name__ == "__main__":
    main()