#Importing necessary libraries
import streamlit as st 
import numpy as np 
import joblib  
import json 
from stable_baselines3 import PPO, SAC
from scipy.stats import mode  
import gym  
from gym import spaces  
import warnings  
import pandas as pd  
import math  
import logging  
import os  
from sklearn.covariance import LedoitWolf  
import alpaca_trade_api as tradeapi  
import plotly.express as px  
import matplotlib.pyplot as plt  
import time  
warnings.filterwarnings('ignore')  

#Configuring logging to capture info and error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Helper function to normalize ticker symbols for Alpaca API compatibility
def normalize_ticker(ticker):
    """Normalize ticker symbols by removing hyphens and slashes."""
    return ticker.replace("-", "").replace("/", "")

#Loading data and models for a specific agent (stock, crypto, or ETF)
def load_agent_data(agent_name):
    """Load agent-specific data and models from files.
    
    Args:
        agent_name (str): Name of the agent ('stock', 'crypto', or 'etf')
    
    Returns:
        tuple: Tickers, prices, regimes, micro/macro indicators, PPO model, and scalers
    """
    logger.info(f"Loading data for {agent_name}...")
    #Listting of required files for the agent
    files = [
        f"{agent_name}_tickers.json",
        f"prices_{agent_name}.npy",
        f"regime_{agent_name}.npy",
        f"micro_indicators_{agent_name}.npy",
        f"macro_indicators_{agent_name}.npy",
        f"ppo_{agent_name}_model.zip",
        f"indicator_scaler_{agent_name}.pkl",
        f"macro_scaler_{agent_name}.pkl"
    ]
    #Checking if all required files exist
    for file in files:
        if not os.path.exists(file):
            logger.error(f"Missing file for {agent_name}: {file}")
            raise FileNotFoundError(f"Missing file for {agent_name}: {file}")
    
    #Loading ticker list from JSON
    logger.info(f"Loading {agent_name}_tickers.json")
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)
    
    #Loading price data and validate
    logger.info(f"Loading prices_{agent_name}.npy")
    prices = np.load(f"prices_{agent_name}.npy")
    if np.any(prices <= 0) or np.any(np.isnan(prices)):
        logger.error(f"Invalid prices for {agent_name}: contains negative or NaN values")
        raise ValueError(f"Invalid prices for {agent_name}: contains negative or NaN values")
    
    #Loading regime, micro, and macro indicators
    logger.info(f"Loading regime_{agent_name}.npy")
    regimes = np.load(f"regime_{agent_name}.npy")
    logger.info(f"Loading micro_indicators_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    logger.info(f"Loading macro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")
    
    #Loading PPO model and scalers
    logger.info(f"Loading ppo_{agent_name}_model.zip")
    model = PPO.load(f"ppo_{agent_name}_model.zip")
    logger.info(f"Loading indicator_scaler_{agent_name}.pkl")
    scaler_indicator = joblib.load(f"indicator_scaler_{agent_name}.pkl")
    logger.info(f"Loading macro_scaler_{agent_name}.pkl")
    scaler_macro = joblib.load(f"macro_scaler_{agent_name}.pkl")
    
    logger.info(f"Data for {agent_name} loaded successfully")
    return tickers, prices, regimes, micro_indicators, macro_indicators, model, scaler_indicator, scaler_macro

#Custom Gym environment for PPO agents
class AgentEnv(gym.Env):
    """Gym environment for individual PPO agents managing a portfolio."""
    def __init__(self, prices, regimes, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001):
        """Initialize the environment with price data and indicators.
        
        Args:
            prices (np.ndarray): Historical price data for assets
            regimes (np.ndarray): Market regime indicators
            micro_indicators (np.ndarray): Micro-level indicators
            macro_indicators (np.ndarray): Macro-level indicators
            initial_amount (float): Initial portfolio value
            risk_appetite (float): Risk preference (0 to 1)
            transaction_fee (float): Fee per transaction
        """
        super(AgentEnv, self).__init__()
        self.prices = prices
        self.regimes = regimes
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee
        self.n_assets = prices.shape[1]  #Number of assets
        self.num_regimes = 3  #Number of market regimes
        self.current_step = 0  #Current time step
        #Defining observation space (prices, portfolio value, risk, regimes, indicators)
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1] + macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)  # Portfolio weights

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        return self._next_observation()

    def _next_observation(self):
        """Generate the observation for the current step."""
        prices_part = self.prices[self.current_step] / (self.prices[0] + 1e-12)  #Normalize prices
        value_part = np.array([self.portfolio_value / self.initial_amount])  #Normalize portfolio value
        risk_part = np.array([self.risk_appetite])  #Risk appetite
        regime_onehot = np.zeros(self.num_regimes)  #One-hot encode regime
        regime_onehot[self.regimes[self.current_step]] = 1
        #Combine all observation components
        obs = np.concatenate([
            prices_part,
            value_part,
            risk_part,
            regime_onehot,
            self.micro_indicators[self.current_step],
            self.macro_indicators[self.current_step]
        ])
        return obs.astype(np.float32)

    def get_covariance_and_return(self, window_size=60):
        """Calculate expected returns and covariance matrix for a time window.
        
        Args:
            window_size (int): Number of time steps for calculation
        
        Returns:
            tuple: Expected returns (mu) and covariance matrix (cov)
        """
        step = self.current_step
        start = max(0, step - window_size)
        window_prices = self.prices[start:step + 1]
        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(np.log(window_prices + 1e-12), axis=0)  #Log returns
        mu = np.mean(returns, axis=0)  #Expected returns
        mu = np.clip(mu, -0.001, 0.001)
        cov = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))  #Covariance matrix
        logger.info(f"get_covariance_and_return: mu={mu}, cov_diagonal={np.diag(cov)}")
        return mu, cov

#Extracting portfolio weights and statistics for an agent
def get_agent_portfolio(ppo_model, env, window_size=60):
    """Use PPO model to generate portfolio weights and statistics.
    
    Args:
        ppo_model: Trained PPO model
        env: AgentEnv instance
        window_size (int): Time window for covariance calculation
    
    Returns:
        tuple: Expected returns, covariance matrix, and portfolio weights
    """
    start_time = time.time()  #Start timing for entire function
    env.reset()
    env.current_step = min(window_size, env.prices.shape[0] - 1)  #Set step for recent data
    obs = env._next_observation().astype(np.float32)
    expected_len = int(np.prod(ppo_model.observation_space.shape))
    cur_len = obs.shape[0]
    #Adjusting observation size to match model expectations
    if cur_len < expected_len:
        obs = np.concatenate([obs, np.zeros(expected_len - cur_len, dtype=np.float32)])
    elif cur_len > expected_len:
        obs = obs[:expected_len]
    obs = obs.reshape(1, -1)
    
    #Measuring PPO model prediction time
    ppo_start_time = time.time()
    action, _ = ppo_model.predict(obs, deterministic=True)
    ppo_inference_time = time.time() - ppo_start_time
    logger.info(f"PPO model inference time: {ppo_inference_time:.4f} seconds")
    
    #Normalizing portfolio weights
    weights = action.flatten()
    weights_sum = weights.sum()
    if weights_sum < 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights_sum
    mu, cov = env.get_covariance_and_return(window_size=window_size)  #Get returns and covariance
    
    logger.info(f"Agent Portfolio - mu: {mu}, weights: {weights}")
    total_time = time.time() - start_time
    logger.info(f"Total get_agent_portfolio execution time: {total_time:.4f} seconds")
    
    return mu, cov, weights

#Meta Portfolio Environment for SAC meta-agent
class MetaPortfolioEnv(gym.Env):
    """Gym environment for the SAC meta-agent to optimize portfolio across all assets."""
    def __init__(self, mu, cov, amount, risk, duration, micro_indicators, macro_indicators, regimes, num_regimes=3):
        """Initialize the meta portfolio environment.
        
        Args:
            mu (np.ndarray): Expected returns for all assets
            cov (np.ndarray): Covariance matrix for all assets
            amount (float): Initial investment amount
            risk (float): Risk appetite (0 to 1)
            duration (int): Number of days to simulate
            micro_indicators (np.ndarray): Combined micro indicators
            macro_indicators (np.ndarray): Combined macro indicators
            regimes (np.ndarray): Combined market regimes
            num_regimes (int): Number of market regimes
        """
        super(MetaPortfolioEnv, self).__init__()
        #Regularizing covariance matrix using Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        self.cov = lw.fit(cov).covariance_ if cov.shape[0] > 1 else cov + np.eye(cov.shape[0]) * 1e-6
        self.mu = mu
        self.amount = amount
        self.risk = risk
        self.duration = duration
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.regimes = regimes
        self.num_regimes = num_regimes
        self.n_assets = len(mu)

        #Defining observation space
        obs_dim = self.n_assets * 3 + 1 + self.micro_indicators.shape[1] + self.macro_indicators.shape[1] + self.num_regimes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.current_step = 0
        self.portfolio_value = amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets  #Initial uniform weights
        self.allocation_history = []

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.allocation_history = []
        return self._get_obs()

    def _get_obs(self):
        """Generate observation for the current step."""
        norm_value = self.portfolio_value / self.amount
        regime_onehot = np.zeros(self.num_regimes)
        idx = min(self.current_step, len(self.regimes) - 1)
        regime_onehot[self.regimes[idx]] = 1
        obs = np.concatenate([
            self.mu,
            np.diag(self.cov),
            self.prev_weights,
            [norm_value],
            self.micro_indicators[idx],
            self.macro_indicators[idx],
            regime_onehot
        ])
        return obs.astype(np.float32)

    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action (np.ndarray): Portfolio weights from SAC agent
        
        Returns:
            tuple: Next observation, reward, done flag, and info dictionary
        """
        self.current_step += 1
        weights = np.clip(action, 0, 1)  #Ensuring weights are non-negative
        weights = np.clip(weights, 0, 0.2)  #Capping individual weights at 20%
        total = np.sum(weights)
        if total < 1e-8:
            weights = np.ones_like(weights) / len(weights)  #Uniform weights if sum is near zero
        else:
            weights /= total
        expected_return = np.dot(weights, self.mu)  #Calculating portfolio return
        expected_return = np.clip(expected_return, -0.002, 0.002)
        variance = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))  #Portfolio risk
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + expected_return)  #Updating portfolio value
        allocation = weights * self.portfolio_value
        self.allocation_history.append(allocation)
        log_return = np.log(self.portfolio_value / (prev_value + 1e-12) + 1e-12)  # Log return
        log_return = np.clip(log_return, -0.002, 0.002)
        logger.info(f"Step {self.current_step}: log_return={log_return:.6f}, portfolio_value={self.portfolio_value:.2f}, expected_return={expected_return:.6f}")
        reward = log_return - self.risk * variance  #Reward: return minus risk penalty
        done = self.current_step >= self.duration
        self.prev_weights = weights
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": log_return,
            "risk_penalty": self.risk * variance,
            "reward": reward,
            "variance": variance
        }
        return self._get_obs(), reward, done, info

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

#Helper function to get last prices for order submission
def _get_last_prices_for_tickers(prices_stock, prices_crypto, prices_etf, tickers_stock, tickers_crypto, tickers_etf):
    """Extract last prices for all tickers, handling missing or invalid data.

    Args:
        prices_stock, prices_crypto, prices_etf (np.ndarray): Price data for each asset class
        tickers_stock, tickers_crypto, tickers_etf (list): Ticker lists for each asset class

    Returns:
        tuple: All tickers, last prices, and normalized crypto tickers
    """
    def extract_last_prices(prices, tickers, asset_name):
        if prices is not None and len(prices) > 0 and len(prices[-1]) == len(tickers):
            return list(prices[-1])
        else:
            logger.warning(f"Invalid {asset_name} prices: {len(prices[-1]) if prices is not None and len(prices) > 0 else 0} prices, {len(tickers)} tickers")
            return [0.0] * len(tickers)

    norm_tickers_stock = [normalize_ticker(t) for t in tickers_stock]
    norm_tickers_crypto = [normalize_ticker(t) for t in tickers_crypto]
    norm_tickers_etf = [normalize_ticker(t) for t in tickers_etf]
    all_tickers = norm_tickers_stock + norm_tickers_crypto + norm_tickers_etf

    last_prices = (
        extract_last_prices(prices_stock, tickers_stock, "stock") +
        extract_last_prices(prices_crypto, tickers_crypto, "crypto") +
        extract_last_prices(prices_etf, tickers_etf, "ETF")
    )
    last_prices = np.array(last_prices)

    # Validating ticker-price alignment
    if len(all_tickers) != len(last_prices):
        logger.error(f"Mismatch: {len(all_tickers)} tickers, {len(last_prices)} prices")
        st.error(f"Price data mismatch: {len(all_tickers)} tickers but {len(last_prices)} prices. Some orders may be skipped.")

    return all_tickers, last_prices, norm_tickers_crypto

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "paper_trading_enabled": False,
        "alpaca_api_key": "",
        "alpaca_secret_key": "",
        "alpaca_base_url": "https://paper-api.alpaca.markets",
        "alpaca_client": None,
        "allocation_log": None,
        "all_tickers": None,
        "display_tickers": None,
        "returns": None,
        "summary_df": None,
        "alloc_df": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# --- Alpaca Connection ---
def connect_to_alpaca():
    if tradeapi is None:
        st.sidebar.error("alpaca-trade-api package not installed.")
        return None
    if not st.session_state["alpaca_api_key"] or not st.session_state["alpaca_secret_key"]:
        st.sidebar.error("Please fill both API Key and Secret.")
        return None
    try:
        client = tradeapi.REST(
            st.session_state["alpaca_api_key"],
            st.session_state["alpaca_secret_key"],
            base_url=st.session_state["alpaca_base_url"],
            api_version='v2'
        )
        account = client.get_account()
        st.sidebar.success(f"Connected to Alpaca (status: {account.status}, Buying Power: ${float(account.buying_power):,.2f})")
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {e}")
        return None


# --- Data Loading ---
def load_all_agents():
    try:
        stock_data = load_agent_data("stock")
        crypto_data = load_agent_data("crypto")
        etf_data = load_agent_data("etf")

        tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock, _, _ = stock_data
        tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto, _, _ = crypto_data
        tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf, _, _ = etf_data

        display_tickers = tickers_stock + tickers_crypto + tickers_etf

        tickers_stock = [normalize_ticker(t) for t in tickers_stock]
        tickers_crypto = [normalize_ticker(t) for t in tickers_crypto]
        tickers_etf = [normalize_ticker(t) for t in tickers_etf]

        return {
            "stock": (tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock),
            "crypto": (tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto),
            "etf": (tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf),
            "display_tickers": display_tickers
        }
    except Exception as e:
        st.error(f"Error loading models/data: {e}")
        return None


# --- SAC Meta Model Loading ---
def load_sac_model():
    try:
        if not os.path.exists("sac_meta_model.zip"):
            raise FileNotFoundError("SAC meta model file not found")
        return SAC.load("sac_meta_model.zip")
    except Exception as e:
        st.error(f"Failed to load SAC meta model: {e}")
        return None


# --- Run Allocation Simulation ---
def run_allocation(amount, risk, duration, agents_data):
    # ... put your current allocation logic here, return summary_df, alloc_df, allocation_log
    pass


# --- Display Results ---
def display_results(summary_df, alloc_df):
    if summary_df is not None:
        st.subheader("Meta Portfolio Allocation Summary")
        st.table(summary_df)
    if alloc_df is not None:
        st.subheader("Asset Allocations on Final Day")
        df_display = alloc_df.copy()
        df_display["Allocation ($)"] = df_display["Allocation ($)"].map("${:,.2f}".format)
        df_display["Allocation (%)"] = df_display["Allocation (%)"].map("{:.2f}%".format)
        st.dataframe(df_display)
    # ... pie charts, trends plots, logs, download buttons here


# --- Submit Orders to Alpaca ---
def submit_orders():
    # ... put your order submission logic here
    pass

def main():
    st.set_page_config(page_title="Meta Portfolio + Alpaca Paper Trading", layout="wide")
    st.title("Meta Portfolio Allocation with SAC Meta-Agent + Alpaca Paper Trading")

    init_session_state()

    # Sidebar - Paper trading
    st.sidebar.header("Paper Trading (Alpaca)")
    st.session_state["paper_trading_enabled"] = st.sidebar.checkbox("Enable Paper Trading", value=st.session_state["paper_trading_enabled"])

    if st.session_state["paper_trading_enabled"]:
        st.sidebar.text_input("API Key", key="alpaca_api_key")
        st.sidebar.text_input("Secret Key", type="password", key="alpaca_secret_key")
        st.sidebar.text_input("Base URL", key="alpaca_base_url")

        if st.sidebar.button("Save & Connect to Alpaca"):
            st.session_state["alpaca_client"] = connect_to_alpaca()

        if st.sidebar.button("Reset Session State"):
            st.session_state.clear()
            st.sidebar.success("Session reset.")
    else:
        st.session_state["alpaca_client"] = None

    # Load agents data
    agents_data = load_all_agents()
    if agents_data:
        st.session_state["display_tickers"] = agents_data["display_tickers"]

    # Inputs
    amount = st.number_input("Total Investment Amount ($)", min_value=1000.0, value=10000.0, step=1000.0)
    risk = st.slider("Risk Appetite", 0.0, 1.0, 0.5, 0.01)
    duration = st.number_input("Investment Duration (days)", min_value=1, value=30)

    # Run allocation
    if st.button("Run Portfolio Allocation") and agents_data:
        model = load_sac_model()
        if model:
            summary_df, alloc_df, allocation_log = run_allocation(amount, risk, duration, agents_data)
            st.session_state["summary_df"] = summary_df
            st.session_state["alloc_df"] = alloc_df
            st.session_state["allocation_log"] = allocation_log

    # Display results
    display_results(st.session_state["summary_df"], st.session_state["alloc_df"])

    # Order submission
    st.sidebar.header("Submit Orders")
    if st.sidebar.button("Submit Paper Orders (final allocation)"):
        submit_orders()


if __name__ == "__main__":
    main()