#Importing libraries
import ta.momentum
import ta.trend
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
#Technical Analysis library
import ta

#Downloading historical Data
#We can remove this part when we have seperate data file from API

#Defining tickers and date range
tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT']
start_date = '2015-01-01'
end_date = '2025-05-20'

#Downloading adjusted price for stocks from yahoo finance library
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
adj_close_data = pd.DataFrame()

#Looping through each tickers
for ticker in tickers:
    adj_close_data[ticker] = data[ticker]['Close']

#Downloading VIX index as a proxy for market volatiliy indicators (Regime indicator)
vix = yf.download('^VIX', start=start_date, end=end_date)['Close']

#Aligning stock and VIX data by date and dropping any missing values
adj_close_data = adj_close_data.loc[vix.index].dropna()
vix = vix.loc[adj_close_data.index]

#Extractin prices as numPy array
prices = adj_close_data.values

#Normalizing VIX values to [0,1] range
vix_norm = (vix - vix.min()) / (vix.max() - vix.min())

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

regime_classes = classify_regime(vix_norm.values)

def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        close = df[col]
        rsi = ta.momentum.RSIIndicator(close).rsi().fillna(50).values
        macd = ta.trend.MACD(close).macd_diff().fillna(0).values
        asset_features = np.vstack([rsi, macd]).T
        features.append(asset_features)
    return np.hstack(features)

micro_indicators = compute_micro_indicators(adj_close_data)

adj_close_data.to_csv("adj_close_prices.csv")
vix_norm_df = pd.DataFrame(regime_classes, index=adj_close_data.index, columns=["Regime Class"])
vix_norm_df.to_csv("regime_data.csv")

micro_cols = []
for ticker in tickers:
    micro_cols.extend([f"{ticker}_RSI", f"{ticker}_MACD"])
micro_df = pd.DataFrame(micro_indicators, index=adj_close_data.index, columns=micro_cols)
micro_df.to_csv("micro_indicators.csv")

def one_hot_regime(regime_class, num_classes=3):
    one_hot = np.zeros(num_classes)
    one_hot[regime_class] = 1
    return one_hot

class StockPortfolioEnv(gym.Env):
    def __init__(self,prices,regime_class,micro_indicators,amount, risk_appetite, duration):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.amount = amount
        self.risk_appetite = risk_appetite
        self.duration = duration
        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.amount
        self.last_allocation = np.ones(self.n_assets) / self.n_assets
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, 0,1)
        action = action / (np.sum(action) + 1e-8)
        prev_prices = self.prices[self.current_step]
        self.current_step += 1
        curr_prices = self.prices[self.current_step]
        returns = (curr_prices - prev_prices) / prev_prices
        portfolio_return = np.dot(action, returns)
        self.portfolio_value *= (1 + portfolio_return)
        risk_penalty = np.std(returns)
        adjusted_reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty
        self.last_allocation = action
        done = self.current_step >= self.duration or self.current_step >= len(self.prices) -1
        return self._get_observation(), adjusted_reward, done, {}

    def _get_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1
        return np.concatenate([
            self.prices[self.current_step],
            [self.portfolio_value],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step]
        ])

initial_amount = 100000
env = StockPortfolioEnv(prices, regime_classes, micro_indicators, amount=initial_amount, risk_appetite=0.5, duration=365)
model = PPO('MlpPolicy', env, verbose=1, device='cpu')
model.learn(total_timesteps=500000)

obs = env.reset()
allocations = []
values = []
transaction_costs = []
actions_log = []

for _ in range(252):
    prev_allocation = getattr(env, 'last_allocation', np.ones(env.n_assets) / env.n_assets)
    prev_value = env.portfolio_value
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    action = np.clip(action, 0, 1)
    action /= (np.sum(action) + 1e-8)
    dollar_allocation = env.portfolio_value * action
    allocation_dict = {ticker: round(dollar_allocation[i], 2) for i, ticker in enumerate(tickers)}
    turnover = np.sum(np.abs(action - prev_allocation))
    transaction_cost = prev_value * turnover * 0.001
    transaction_costs.append(transaction_cost)

    labels = []
    for curr, prev in zip(action, prev_allocation):
        if curr > prev + 1e-4:
            labels.append("Buy")
        elif curr < prev - 1e-4:
            labels.append("Sell")
        else:
            labels.append("Hold")
    actions_log.append(labels)

    allocations.append(allocation_dict)
    values.append(env.portfolio_value)
    if done:
        break

df_alloc = pd.DataFrame(allocations)
df_labels = pd.DataFrame(actions_log, columns=[f"{ticker}_Action" for ticker in tickers])
df_values = pd.DataFrame({"Portfolio_Value": values, "Transaction_Cost": transaction_costs + [0]})
final_log = pd.concat([df_alloc, df_labels, df_values], axis=1)
final_log.to_csv("allocation_with_actions_and_costs.csv", index=False)

print("\n Final Portfolio Allocation and Profit")
total_value = sum(allocations[-1].values())
for ticker, val in allocations[-1].items():
    print(f"{ticker}: ${val:,.2f}")
total_profit = total_value - initial_amount
print(f"\nTotal Profit: ${total_profit:,.2f}")

plt.figure(figsize=(10, 5))
plt.plot(values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Days")
plt.ylabel("Value ($)")
plt.grid(True)
plt.show()
