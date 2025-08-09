"""
Trading environment classes for deep hedging
"""

import numpy as np
import pandas as pd


class DeepHedgingEnvironment:
    """Trading environment for deep hedging.

    State features (length 10):
    [normalized_price, returns, volatility, normalized_option_price,
     delta, scaled_gamma, scaled_theta, hedge_position,
     normalized_portfolio_value, normalized_step]
    """

    def __init__(
        self,
        data,
        option_data,
        transaction_cost=0.0001,
        initial_capital=100000,
        episode_length=60,
        position_scale=1000,
    ):
        self.data = data
        self.option_data = option_data
        # Allow transaction_cost provided either as fraction (e.g., 0.0001) or bps (e.g., 1)
        self.tc_bps = transaction_cost * 10000 if transaction_cost < 1 else transaction_cost
        self.initial_capital = initial_capital
        self.episode_length = episode_length
        self.position_scale = position_scale
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.hedge_position = 0.0
        self.portfolio_value = float(self.initial_capital)
        self.cumulative_pnl = 0.0
        self.daily_pnl = []
        return self._get_state()

    def _get_state(self):
        """Get current state representation."""
        if self.current_step >= len(self.data) - 1:
            return None

        current_data = self.data.iloc[self.current_step]
        current_option = self.option_data.iloc[self.current_step]

        normalized_price = current_data['Close'] / 100.0
        returns = (
            current_data['Returns'] if not pd.isna(current_data['Returns']) else 0.0
        )
        volatility = current_option['volatility']
        normalized_option_price = current_option['option_price'] / 100.0
        delta = current_option['delta']
        scaled_gamma = current_option['gamma'] * 100.0
        scaled_theta = current_option['theta'] / 100.0
        normalized_portfolio_value = self.portfolio_value / self.initial_capital
        normalized_step = self.current_step / float(self.episode_length)

        state = np.array(
            [
                normalized_price,
                returns,
                volatility,
                normalized_option_price,
                delta,
                scaled_gamma,
                scaled_theta,
                self.hedge_position,
                normalized_portfolio_value,
                normalized_step,
            ]
        )

        return state

    def step(self, action):
        """Execute one trading step."""
        if self.current_step >= len(self.data) - 1:
            return None, 0.0, True, {}

        # Current and next market/option points
        current_price = float(self.data.iloc[self.current_step]['Close'])
        current_option = self.option_data.iloc[self.current_step]
        next_price = float(self.data.iloc[self.current_step + 1]['Close'])
        next_option = self.option_data.iloc[self.current_step + 1]

        # Option P&L (mark-to-market change)
        option_pnl = float(next_option['option_price'] - current_option['option_price'])

        # Hedge P&L from holding current position through price move
        hedge_pnl = -float(self.hedge_position) * (next_price - current_price)

        # Transaction costs on notional traded: |Δposition| * price * (bps/10000)
        new_position = float(action) * float(self.position_scale)
        position_change = new_position - float(self.hedge_position)
        tc_rate = float(self.tc_bps) / 10000.0
        transaction_costs = abs(position_change) * current_price * tc_rate

        # Update portfolio and bookkeeping
        total_pnl = option_pnl + hedge_pnl - transaction_costs
        self.portfolio_value += total_pnl
        self.cumulative_pnl += total_pnl
        self.daily_pnl.append(total_pnl)
        self.hedge_position = new_position

        # Advance step
        self.current_step += 1
        done = (
            self.current_step >= len(self.data) - 1
            or self.current_step >= self.episode_length
        )

        # Reward: negative P&L (risk minimization)
        reward = -total_pnl
        next_state = self._get_state()

        return next_state, reward, done, {
            'option_pnl': option_pnl,
            'hedge_pnl': hedge_pnl,
            'transaction_costs': transaction_costs,
            'total_pnl': total_pnl,
            'portfolio_value': self.portfolio_value,
        }

    # Remove _calculate_cvar as it's no longer used for immediate reward