# Importing libraries
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
# Setting up the environment

# List of stock tickers to analyze
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "UNH"
]

# Define the start and end dates for historical data
start_date = '2015-01-01'
end_date = '2025-05-20'

# Downloading adjusted stock prices
print("Downloading stock data...")
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
adj_close_data = pd.DataFrame()
for ticker in tickers:
    adj_close_data[ticker] = data[ticker]['Close']

# Downloading VIX index as a market regime proxy
print("Downloading VIX data...")
vix = yf.download('^VIX', start=start_date, end=end_date)['Close']

# Align stock and VIX data by index
adj_close_data = adj_close_data.loc[vix.index].dropna()
vix = vix.loc[adj_close_data.index]

# Normalize VIX for regime classification
scaler = MinMaxScaler()
vix_norm = scaler.fit_transform(vix.values.reshape(-1, 1)).flatten()

def classify_regime(vix_norm):
    bullish_thresh = 0.33
    bearish_thresh = 0.66
    regimes_class = []
    for val in vix_norm:
        if val <= bullish_thresh:
            regimes_class.append(0)  # Bullish
        elif val <= bearish_thresh:
            regimes_class.append(1)  # Sideways
        else:
            regimes_class.append(2)  # Bearish
    return np.array(regimes_class)

regime_classes = classify_regime(vix_norm)

# Function to compute technical indicators for each asset
def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        close = df[col]
        high = df[col]
        low = df[col]

        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        acc_dist = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=df[col]).acc_dist_index().fillna(0).values
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().fillna(method='backfill').values
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().fillna(method='backfill').values - ichimoku.ichimoku_base_line().fillna(method='backfill').values
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        ultimate_osc = ta.momentum.UltimateOscillator(high=high, low=low, close=close).ultimate_oscillator().fillna(50).values
        ppo = ta.momentum.PercentagePriceOscillator(close=close).ppo().fillna(0).values
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().fillna(method='backfill').values
        keltner = ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_hband().fillna(method='backfill').values - \
                  ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_lband().fillna(method='backfill').values
        ema = ta.trend.EMAIndicator(close=close, window=10).ema_indicator().fillna(method='backfill').values
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        asset_features = np.vstack([
            rsi, macd_diff, ema, bb_width, stoch_k,
            obv, acc_dist,
            adx, psar, ichimoku_diff,
            williams_r, ultimate_osc, ppo,
            atr, keltner
        ]).T
        features.append(asset_features)

    return np.hstack(features)

# Compute and normalize indicators
print("Computing technical indicators...")
micro_indicators = compute_micro_indicators(adj_close_data)
micro_indicators = MinMaxScaler().fit_transform(micro_indicators)
prices = adj_close_data.values

# Custom OpenAI Gym environment
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, 
                 regime_class, 
                 micro_indicators, 
                 initial_amount=10000, 
                 risk_appetite=0.2, 
                 transaction_fee=0.001):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1
        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],
            [self.portfolio_value / self.initial_amount],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step]
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        transaction_cost = np.sum(np.abs(trades)) * self.transaction_fee
        return transaction_cost, target_dollars

    def step(self, action):
        action = np.clip(action, 0, 1)
        new_weights = action / (np.sum(action) + 1e-8)
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost
        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices

        portfolio_return = np.dot(new_weights, returns)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        risk_penalty = np.std(returns)
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value)

        self.current_step += 1
        self.portfolio_value = new_portfolio_value
        done = self.current_step >= len(self.prices) - 2

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        current_prices = self.prices[self.current_step]
        holdings_value = self.current_holdings * current_prices
        total_value = np.sum(holdings_value)
        print(f"\nStep: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {self.current_holdings[i]:.2f} shares (${holdings_value[i]:.2f}, {100*holdings_value[i]/total_value:.1f}%)")

# Create environment
print("Creating environment...")
env = StockPortfolioEnv(prices, regime_classes, micro_indicators, 10000, 0.5, 0.001)

# Train PPO model
print("Training model...")
model = PPO('MlpPolicy', 
            env, 
            verbose=1, 
            learning_rate=3e-4, 
            n_steps=2048, 
            batch_size=64, 
            n_epochs=10, 
            gamma=0.99, 
            gae_lambda=0.95, 
            clip_range=0.2, 
            ent_coef=0.01)
model.learn(total_timesteps=1000000)

# Save trained PPO model
print("Saving model and scalers...")
model.save("ppo_stock_model")

# Save VIX scaler

joblib.dump(scaler, "vix_scaler.pkl")

# Save ticker list

with open("tickers.json", "w") as f:
    json.dump(tickers, f)

# Optionally, save the indicator scaler if needed later
indicator_scaler = MinMaxScaler().fit(micro_indicators)
micro_indicators = indicator_scaler.transform(micro_indicators)
joblib.dump(indicator_scaler, "indicator_scaler.pkl")


# Test model
print("Testing model...")
obs = env.reset()
done = False
portfolio_values = [env.initial_amount]
allocations = []
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    portfolio_values.append(env.portfolio_value)
    allocations.append(env.asset_weights)
    if done or len(portfolio_values) % 50 == 0:
        env.render()

# Print results
print("\n=== Final Results ===")
print(f"Initial Investment: ${env.initial_amount:,.2f}")
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Profit: ${portfolio_values[-1] - env.initial_amount:,.2f}")
print(f"Annualized Return: {100 * ((portfolio_values[-1] / env.initial_amount) ** (252 / len(portfolio_values)) - 1):.2f}%")

# Asset performance
initial_prices = prices[0]
final_prices = prices[-1]
print("\nIndividual Asset Performance:")
for i, ticker in enumerate(tickers):
    asset_return = 100 * (final_prices[i] - initial_prices[i]) / initial_prices[i]
    print(f"{ticker}: {asset_return:.2f}%")

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time (With Transaction Fees)")
plt.xlabel("Days")
plt.ylabel("Value ($)")
plt.grid(True)
plt.show()

# Plot asset allocations
allocations = np.array(allocations)
plt.figure(figsize=(12, 6))
for i in range(len(tickers)):
    plt.plot(allocations[:, i], label=tickers[i])
plt.title("Asset Allocation Over Time")
plt.xlabel("Days")
plt.ylabel("Allocation Weight")
plt.legend()
plt.grid(True)
plt.show()

# Plot regime classification
plt.figure(figsize=(12, 6))
plt.plot(vix_norm, label='Normalized VIX')
plt.plot([0.33]*len(vix_norm), 'g--', label='Bullish Threshold')
plt.plot([0.66]*len(vix_norm), 'r--', label='Bearish Threshold')
plt.title("Market Regime Classification")
plt.xlabel("Days")
plt.ylabel("VIX (Normalized)")
plt.legend()
plt.grid(True)
plt.show()
