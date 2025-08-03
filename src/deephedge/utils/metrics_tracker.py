"""
Metrics tracking utilities for deep hedging experiments
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import torch


class MetricsTracker:
    """Comprehensive metrics tracking for deep hedging experiments"""
    
    def __init__(self, config, project_name="deep_hedging", entity=None):
        """
        Initialize metrics tracker
        
        Args:
            config: Experiment configuration
            project_name: W&B project name
            entity: W&B entity/username
        """
        self.config = config
        self.project_name = project_name
        self.entity = entity
        
        # Initialize W&B
        self.wandb_run = None
        self._init_wandb()
        
        # Metrics storage
        self.gan_metrics = []
        self.hedger_metrics = []
        self.evaluation_metrics = {}
        
    def _init_wandb(self):
        """Initialize Weights & Biases run"""
        try:
            self.wandb_run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=self.config.to_container(),
                name=f"{self.config.experiment.name}_{self.config.experiment.seed}",
                tags=["deep_hedging", "gan", "actor_critic"],
                notes=f"Deep hedging experiment with GAN-generated worst-case paths"
            )
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")
            self.wandb_run = None
    
    def log_gan_metrics(self, epoch, d_loss, g_loss, real_score, fake_score):
        """Log GAN training metrics"""
        metrics = {
            "gan/d_loss": d_loss,
            "gan/g_loss": g_loss,
            "gan/real_score": real_score,
            "gan/fake_score": fake_score,
            "gan/epoch": epoch
        }
        
        self.gan_metrics.append(metrics)
        
        if self.wandb_run:
            wandb.log(metrics, step=epoch)
    
    def log_hedger_metrics(self, episode, avg_reward, total_pnl, cvar_95, turnover):
        """Log hedger training metrics"""
        metrics = {
            "hedger/avg_reward": avg_reward,
            "hedger/total_pnl": total_pnl,
            "hedger/cvar_95": cvar_95,
            "hedger/turnover": turnover,
            "hedger/episode": episode
        }
        
        self.hedger_metrics.append(metrics)
        
        if self.wandb_run:
            wandb.log(metrics, step=episode)
    
    def log_evaluation_results(self, results, crash_periods):
        """Log evaluation results"""
        for period in crash_periods:
            ai_key = f"{period}_AI"
            delta_key = f"{period}_Delta"
            
            if ai_key in results and delta_key in results:
                ai_data = results[ai_key]
                delta_data = results[delta_key]
                
                # Calculate improvement
                cvar_improvement = ((delta_data['cvar_95'] - ai_data['cvar_95']) / delta_data['cvar_95']) * 100
                
                metrics = {
                    f"evaluation/{period}/ai_cvar_95": ai_data['cvar_95'],
                    f"evaluation/{period}/delta_cvar_95": delta_data['cvar_95'],
                    f"evaluation/{period}/ai_mean_pnl": ai_data['mean_pnl'],
                    f"evaluation/{period}/delta_mean_pnl": delta_data['mean_pnl'],
                    f"evaluation/{period}/cvar_improvement": cvar_improvement
                }
                
                self.evaluation_metrics[period] = metrics
                
                if self.wandb_run:
                    wandb.log(metrics)
    
    def create_loss_curves(self, save_path="runs"):
        """Create and save loss curves"""
        if not self.gan_metrics and not self.hedger_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GAN Loss Curves
        if self.gan_metrics:
            epochs = [m["gan/epoch"] for m in self.gan_metrics]
            d_losses = [m["gan/d_loss"] for m in self.gan_metrics]
            g_losses = [m["gan/g_loss"] for m in self.gan_metrics]
            
            axes[0, 0].plot(epochs, d_losses, label='Discriminator Loss', color='red')
            axes[0, 0].plot(epochs, g_losses, label='Generator Loss', color='blue')
            axes[0, 0].set_title('GAN Training Losses')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # GAN Scores
            real_scores = [m["gan/real_score"] for m in self.gan_metrics]
            fake_scores = [m["gan/fake_score"] for m in self.gan_metrics]
            
            axes[0, 1].plot(epochs, real_scores, label='Real Score', color='green')
            axes[0, 1].plot(epochs, fake_scores, label='Fake Score', color='orange')
            axes[0, 1].set_title('GAN Discriminator Scores')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Hedger Metrics
        if self.hedger_metrics:
            episodes = [m["hedger/episode"] for m in self.hedger_metrics]
            rewards = [m["hedger/avg_reward"] for m in self.hedger_metrics]
            cvars = [m["hedger/cvar_95"] for m in self.hedger_metrics]
            
            axes[1, 0].plot(episodes, rewards, label='Avg Reward', color='purple')
            axes[1, 0].set_title('Hedger Training Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(episodes, cvars, label='CVaR 95%', color='brown')
            axes[1, 1].set_title('Hedger CVaR 95%')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('CVaR 95%')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        plot_file = save_path / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # Log to W&B
        if self.wandb_run:
            wandb.log({"training_curves": wandb.Image(plot_file)})
        
        plt.close()
    
    def create_cvar_comparison_chart(self, results, crash_periods, save_path="runs"):
        """Create CVaR comparison chart"""
        if not results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CVaR Comparison
        periods = []
        ai_cvars = []
        delta_cvars = []
        improvements = []
        
        for period in crash_periods:
            ai_key = f"{period}_AI"
            delta_key = f"{period}_Delta"
            
            if ai_key in results and delta_key in results:
                ai_data = results[ai_key]
                delta_data = results[delta_key]
                
                periods.append(period)
                ai_cvars.append(ai_data['cvar_95'])
                delta_cvars.append(delta_data['cvar_95'])
                
                improvement = ((delta_data['cvar_95'] - ai_data['cvar_95']) / delta_data['cvar_95']) * 100
                improvements.append(improvement)
        
        if periods:
            x = np.arange(len(periods))
            width = 0.35
            
            axes[0].bar(x - width/2, ai_cvars, width, label='AI Hedger', alpha=0.8, color='blue')
            axes[0].bar(x + width/2, delta_cvars, width, label='Delta Hedger', alpha=0.8, color='red')
            axes[0].set_title('CVaR 95% Comparison')
            axes[0].set_ylabel('CVaR 95%')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(periods)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Improvement Chart
            axes[1].bar(x, improvements, color='green', alpha=0.8)
            axes[1].set_title('CVaR Improvement (%)')
            axes[1].set_ylabel('Improvement (%)')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(periods)
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(improvements):
                axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        plot_file = save_path / "cvar_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # Log to W&B
        if self.wandb_run:
            wandb.log({"cvar_comparison": wandb.Image(plot_file)})
        
        plt.close()
    
    def log_pnl_traces(self, pnl_series, period_name):
        """Log P&L traces to W&B"""
        if self.wandb_run and pnl_series:
            # Create P&L trace plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(pnl_series, label=f'{period_name} P&L', alpha=0.7)
            ax.set_title(f'P&L Trace - {period_name}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('P&L')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log to W&B
            wandb.log({f"pnl_trace_{period_name}": wandb.Image(fig)})
            plt.close()
    
    def finish(self):
        """Finish the W&B run"""
        if self.wandb_run:
            wandb.finish()
    
    def get_summary_metrics(self):
        """Get summary of all metrics"""
        summary = {
            "gan_metrics_count": len(self.gan_metrics),
            "hedger_metrics_count": len(self.hedger_metrics),
            "evaluation_periods": len(self.evaluation_metrics)
        }
        
        if self.gan_metrics:
            summary["final_gan_d_loss"] = self.gan_metrics[-1]["gan/d_loss"]
            summary["final_gan_g_loss"] = self.gan_metrics[-1]["gan/g_loss"]
        
        if self.hedger_metrics:
            summary["final_hedger_reward"] = self.hedger_metrics[-1]["hedger/avg_reward"]
            summary["final_hedger_cvar"] = self.hedger_metrics[-1]["hedger/cvar_95"]
        
        return summary 