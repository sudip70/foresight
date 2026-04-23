"""
Crypto Portfolio PPO Training Environment
------------------------------------------
This script defines:
- A Gym environment for cryptocurrency portfolio optimization.
- Data fetching and preprocessing functions.
- PPO model training workflow.
"""

import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib
import json
from gym.utils import seeding

# Disable unnecessary warnings
warnings.filterwarnings('ignore')

# ==============================
# CONSTANTS
# ==============================
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "LTC-USD", "DOT-USD", "AVAX-USD"
]
START_DATE = '2018-01-01'
END_DATE = '2025-05-20'
SELECTED_MACRO_COLUMNS = [
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator"
]
NUM_REGIMES = 3
BULLISH_THRESH = 0.33
BEARISH_THRESH = 0.66


# ==============================
# FEATURE ENGINEERING FUNCTIONS
# ==============================
def compute_micro_indicators(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates technical indicators for each asset.

    Args:
        df (pd.DataFrame): Closing price DataFrame.

    Returns:
        np.ndarray: Stacked features for all assets.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if df.shape[1] == 0:
        raise ValueError("No assets provided")
    if np.any(df <= 0):
        raise ValueError("Prices contain non-positive values")

    features = []
    for col in df.columns:
        close = df[col]
        high = df[col]
        low = df[col]

        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high, low, close).psar().bfill().values
        ichimoku_diff = (ta.trend.IchimokuIndicator(high, low)
                         .ichimoku_conversion_line().bfill().values
                         - ta.trend.IchimokuIndicator(high, low)
                         .ichimoku_base_line().bfill().values)
        rsi = ta.momentum.RSIIndicator(close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high, low, close).williams_r().fillna(-50).values
        stoch_k = ta.momentum.StochasticOscillator(high, low, close).stoch().fillna(0).values
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().bfill().values
        bb_width = ta.volatility.BollingerBands(close).bollinger_wband().fillna(0).values

        asset_features = np.vstack([
            rsi, macd_diff, bb_width, stoch_k,
            obv, adx, psar, ichimoku_diff,
            williams_r, atr
        ]).T
        features.append(asset_features)

    return np.hstack(features)


def classify_regime(btc_norm: np.ndarray) -> np.ndarray:
    """
    Classifies BTC volatility regimes.

    Args:
        btc_norm (np.ndarray): Normalized BTC prices.

    Returns:
        np.ndarray: Regime classification (0,1,2).
    """
    if btc_norm.size == 0:
        return np.array([])

    regimes = []
    for val in btc_norm:
        if val <= BULLISH_THRESH:
            regimes.append(0)
        elif val <= BEARISH_THRESH:
            regimes.append(1)
        else:
            regimes.append(2)
    return np.array(regimes)


# ==============================
# GYM ENVIRONMENT
# ==============================
class CryptoPortfolioEnv(gym.Env):
    """Gym environment for cryptocurrency portfolio optimization."""
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, regime_class, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super().__init__()
        self._validate_inputs(prices, regime_class, micro_indicators, macro_indicators)

        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        obs_shape = self._calculate_obs_shape()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def _validate_inputs(self, prices, regime_class, micro_indicators, macro_indicators):
        """Validate input array shapes."""
        if prices.shape[0] < 2:
            raise ValueError("Prices array must have at least 2 rows")
        if prices.shape[1] == 0:
            raise ValueError("No assets in prices array")
        if regime_class.shape[0] != prices.shape[0]:
            raise ValueError("Regime class length mismatch")
        if micro_indicators.shape[0] != prices.shape[0] or macro_indicators.shape[0] != prices.shape[0]:
            raise ValueError("Indicator arrays must match prices length")

    def _calculate_obs_shape(self):
        """Calculate size of observation vector."""
        return (
            self.n_assets + 2 + NUM_REGIMES +
            self.micro_indicators.shape[1] +
            self.macro_indicators.shape[1]
        )

    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        """Build current observation."""
        regime_onehot = np.zeros(NUM_REGIMES)
        regime_onehot[self.regime_class[self.current_step]] = 1

        return np.concatenate([
            self.prices[self.current_step] / self.prices[0],
            [self.portfolio_value / self.initial_amount],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step],
            self.macro_indicators[self.current_step]
        ])

    def _calculate_transaction_cost(self, new_weights):
        """Calculate transaction costs for rebalancing."""
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        return np.sum(np.abs(trades)) * self.transaction_fee, target_dollars

    def step(self, action):
        """Execute one time step."""
        if self.current_step >= len(self.prices) - 1:
            raise ValueError("End of data reached")

        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        new_weights = action / (np.sum(action) + 1e-8)

        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost

        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        returns = (self.prices[self.current_step + 1] - self.prices[self.current_step]) / self.prices[self.current_step]
        portfolio_return = np.dot(new_weights, returns)
        risk_penalty = np.std(returns) if len(returns) > 1 else 0.0
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value if self.portfolio_value > 0 else 0)

        self.current_step += 1
        self.portfolio_value *= (1 + portfolio_return)
        done = self.current_step >= len(self.prices) - 2

        return self._next_observation(), reward, done, {}

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed_val = seeding.np_random(seed)
        return [seed_val]

    def get_covariance_and_return(self, window_size=20):
        """Calculate rolling mean returns and covariance matrix."""
        window_start = max(0, self.current_step - window_size)
        window_prices = self.prices[window_start:self.current_step + 1]
        if window_prices.shape[0] <= 1:
            return np.zeros(self.n_assets), np.zeros((self.n_assets, self.n_assets))

        returns = np.diff(window_prices, axis=0) / window_prices[:-1]
        return np.mean(returns, axis=0), np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    """Main execution block for training PPO agent."""
    print("Downloading crypto data...")
    data = yf.download(CRYPTO_TICKERS, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=True)
    if data.empty:
        raise ValueError("No data downloaded for tickers and date range.")

    adj_close_data = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in CRYPTO_TICKERS})

    print("Using BTC for volatility regime estimation...")
    scaler = MinMaxScaler()
    btc_norm = scaler.fit_transform(adj_close_data["BTC-USD"].values.reshape(-1, 1)).flatten()
    regime_classes = classify_regime(btc_norm)

    print("Computing indicators...")
    micro_indicators = compute_micro_indicators(adj_close_data)
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = np.nan_to_num(indicator_scaler.transform(micro_indicators))

    print("Processing macroeconomic data...")
    macro_df = pd.read_csv('macroeconomic_data_2010_2024.csv', parse_dates=['Date'], index_col='Date').reindex(adj_close_data.index, method='ffill')
    if not all(col in macro_df.columns for col in SELECTED_MACRO_COLUMNS):
        raise ValueError("Missing required macroeconomic columns")
    macro_df = macro_df[SELECTED_MACRO_COLUMNS]
    macro_scaler = MinMaxScaler().fit(macro_df)
    macro_indicators = macro_scaler.transform(macro_df)

    adj_close_data = adj_close_data.replace([np.inf, -np.inf], np.nan).dropna()
    prices = adj_close_data.to_numpy()
    if prices.shape[0] < 2:
        raise ValueError("Insufficient price data after cleaning")

    np.save("prices_crypto.npy", prices)
    np.save("regime_crypto.npy", regime_classes)
    np.save("micro_indicators_crypto.npy", micro_indicators)
    np.save("macro_indicators_crypto.npy", macro_indicators)

    print("Creating environment...")
    env = CryptoPortfolioEnv(prices, regime_classes, micro_indicators, macro_indicators, 10000, 0.5, 0.001)
    env.seed(42)

    print("Training PPO model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)
    model.learn(total_timesteps=500000, progress_bar=True)

    mu, cov = env.get_covariance_and_return()
    print("Mean expected returns:", mu)
    print("Covariance matrix shape:", cov.shape)

    print("Saving model and scalers...")
    model.save("ppo_crypto_model")
    joblib.dump(scaler, "btc_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler_crypto.pkl")
    joblib.dump(macro_scaler, "macro_scaler_crypto.pkl")
    with open("crypto_tickers.json", "w") as f:
        json.dump(CRYPTO_TICKERS, f)


if __name__ == "__main__":
    main()
