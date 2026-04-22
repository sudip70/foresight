# Importing libraries
import ta.momentum
import ta.trend
import ta.volatility
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler

# Downloading historical Data
tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT']
start_date = '2015-01-01'
end_date = '2025-05-20'

# Downloading adjusted price for stocks from yahoo finance library
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
adj_close_data = pd.DataFrame()

# Looping through each ticker to get Close prices
for ticker in tickers:
    adj_close_data[ticker] = data[ticker]['Close']

# Downloading VIX index as a proxy for market volatility indicators (Regime indicator)
vix = yf.download('^VIX', start=start_date, end=end_date)['Close']

# Aligning stock and VIX data by date and dropping any missing values
adj_close_data = adj_close_data.loc[vix.index].dropna()
vix = vix.loc[adj_close_data.index]

# Extracting prices as numpy array
prices = adj_close_data.values

# Computing market regimes
scaler = MinMaxScaler()
vix_norm = scaler.fit_transform(vix.values.reshape(-1, 1)).flatten()

# Function to define market regimes
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

# Regime classes
regime_classes = classify_regime(vix_norm)

# Computing micro indicators
def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        close = df[col]
        # RSI
        rsi = ta.momentum.RSIIndicator(close).rsi().fillna(50).values
        # MACD diff
        macd = ta.trend.MACD(close).macd_diff().fillna(0).values
        # EMA (window 10)
        ema = ta.trend.EMAIndicator(close, window=10).ema_indicator().fillna(method='backfill').values
        # Bollinger Bands width
        bb_width = ta.volatility.BollingerBands(close).bollinger_wband().fillna(0).values
        # Stochastic Oscillator %K
        stoch_k = ta.momentum.StochasticOscillator(
            high=close, low=close, close=close, window=14, smooth_window=3
        ).stoch().fillna(0).values

        # Stack all indicators vertically for this asset
        asset_features = np.vstack([rsi, macd, ema, bb_width, stoch_k]).T
        features.append(asset_features)

    # Concatenate horizontally all asset features
    return np.hstack(features)

micro_indicators = compute_micro_indicators(adj_close_data)

# Normalize micro indicators
scaler = MinMaxScaler()
micro_indicators = scaler.fit_transform(micro_indicators)

# Custom OpenAI Gym environment for stock portfolio management
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, prices, regime_class, micro_indicators, initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee  # 0.1% transaction fee
        
        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)  # Track number of shares held
        
        # Action space: allocation weight for each asset (sum to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation space: prices + portfolio_value + risk_appetite + regime_onehot + micro_indicators
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets  # Equal weights initially
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()
    
    def _next_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1
        
        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],  # Normalized prices
            [self.portfolio_value / self.initial_amount],  # Normalized portfolio value
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step]
        ])
        return obs
    
    def _calculate_transaction_cost(self, new_weights):
        # Calculate target dollar amounts
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        
        # Calculate trades needed
        trades = target_dollars - current_dollars
        
        # Calculate transaction costs (absolute value because we pay fee on both buys and sells)
        transaction_cost = np.sum(np.abs(trades)) * self.transaction_fee
        
        return transaction_cost, target_dollars
    
    def step(self, action):
        # Normalize action weights to sum to 1
        action = np.clip(action, 0, 1)
        new_weights = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction costs and adjust portfolio value
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost
        
        # Recalculate weights after transaction costs
        if self.portfolio_value > 0:
            new_weights = target_dollars / self.portfolio_value
        else:
            new_weights = np.zeros(self.n_assets)
        
        # Update holdings based on new weights
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights
        
        # Calculate returns for next step
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices
        
        # Update portfolio value based on returns
        portfolio_return = np.dot(new_weights, returns)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        
        # Calculate reward with risk penalty
        risk_penalty = np.std(returns)
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value)
        
        # Update step
        self.current_step += 1
        self.portfolio_value = new_portfolio_value
        
        # Check if done
        done = self.current_step >= len(self.prices) - 2
        
        return self._next_observation(), reward, done, {}
    
    def render(self, mode='human', close=False):
        current_prices = self.prices[self.current_step]
        holdings_value = self.current_holdings * current_prices
        total_value = np.sum(holdings_value)
        
        print(f'\nStep: {self.current_step}')
        print(f'Portfolio Value: ${self.portfolio_value:.2f}')
        print('Current Holdings:')
        for i, ticker in enumerate(tickers):
            print(f'{ticker}: {self.current_holdings[i]:.2f} shares (${holdings_value[i]:.2f}, {100*holdings_value[i]/total_value:.1f}%)')
        print(f'Transaction Fees Paid: ${(self.initial_amount + np.sum(portfolio_values) - self.portfolio_value):.2f}')

# Create environment with transaction fees
env = StockPortfolioEnv(
    prices=prices,
    regime_class=regime_classes,
    micro_indicators=micro_indicators,
    initial_amount=10000,
    risk_appetite=0.2,
    transaction_fee=0.001  # 0.1% transaction fee
)

# Train the model
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(total_timesteps=500000)

# Test the trained model
obs = env.reset()
done = False
portfolio_values = [env.portfolio_value]
allocations = []
transaction_costs = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    # Store portfolio value and allocations
    portfolio_values.append(env.portfolio_value)
    allocations.append(env.asset_weights)
    
    # Render to see detailed allocations
    if done or len(portfolio_values) % 50 == 0:
        env.render()

# Print final results
print("\n=== Final Results ===")
print(f"Initial Investment: ${env.initial_amount:,.2f}")
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Profit: ${portfolio_values[-1] - env.initial_amount:,.2f}")
print(f"Annualized Return: {100 * ((portfolio_values[-1] / env.initial_amount) ** (252 / len(portfolio_values)) - 1):.2f}%")


# Calculate and print individual asset performance
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

# Plot allocations
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