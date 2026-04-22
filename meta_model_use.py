# streamlit_meta_alpaca.py
import streamlit as st
import numpy as np
import joblib
import json
from stable_baselines3 import PPO, SAC
from scipy.stats import mode
import gym
from gym import spaces
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import math
warnings.filterwarnings('ignore')

# Alpaca SDK import (optional)
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

# -------------------------
# Session state initialization (VERY TOP)
# -------------------------
if "paper_trading_enabled" not in st.session_state:
    st.session_state["paper_trading_enabled"] = False
if "alpaca_api_key" not in st.session_state:
    st.session_state["alpaca_api_key"] = ""
if "alpaca_secret_key" not in st.session_state:
    st.session_state["alpaca_secret_key"] = ""
if "alpaca_base_url" not in st.session_state:
    st.session_state["alpaca_base_url"] = "https://paper-api.alpaca.markets"
if "alpaca_client" not in st.session_state:
    st.session_state["alpaca_client"] = None
if "allocation_log" not in st.session_state:
    st.session_state["allocation_log"] = None
if "all_tickers" not in st.session_state:
    st.session_state["all_tickers"] = None

# ===============================
# Load Saved Data and Models
# ===============================
def load_agent_data(agent_name):
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)

    prices = np.load(f"prices_{agent_name}.npy")
    regimes = np.load(f"regime_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")

    model = PPO.load(f"ppo_{agent_name}_model.zip")
    scaler_indicator = joblib.load(f"indicator_scaler_{agent_name}.pkl")
    scaler_macro = joblib.load(f"macro_scaler_{agent_name}.pkl")

    return tickers, prices, regimes, micro_indicators, macro_indicators, model, scaler_indicator, scaler_macro


# ===============================
# Custom Environment for PPO Agents
# ===============================
class AgentEnv(gym.Env):
    def __init__(self, prices, regimes, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001):
        super(AgentEnv, self).__init__()
        self.prices = prices
        self.regimes = regimes
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0

        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1] + macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        return self._next_observation()

    def _next_observation(self):
        # avoid division by zero
        prices_part = self.prices[self.current_step] / (self.prices[0] + 1e-12)
        value_part = np.array([self.portfolio_value / self.initial_amount])
        risk_part = np.array([self.risk_appetite])
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regimes[self.current_step]] = 1

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
        step = self.current_step
        start = max(0, step - window_size)
        window_prices = self.prices[start:step + 1]
        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(np.log(window_prices + 1e-12), axis=0)
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))
        return mu, cov


# ===============================
# Extract Portfolio Info per Agent
# ===============================
def get_agent_portfolio(ppo_model, env, window_size=60):
    env.reset()
    env.current_step = min(window_size, env.prices.shape[0] - 1)  # ensure valid window
    obs = env._next_observation().astype(np.float32)

    expected_len = int(np.prod(ppo_model.observation_space.shape))
    cur_len = obs.shape[0]

    if cur_len < expected_len:
        pad_len = expected_len - cur_len
        obs = np.concatenate([obs, np.zeros(pad_len, dtype=np.float32)])
    elif cur_len > expected_len:
        obs = obs[:expected_len]

    obs = obs.reshape(1, -1)

    action, _ = ppo_model.predict(obs, deterministic=True)
    weights = action.flatten()
    weights_sum = weights.sum()
    if weights_sum < 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights_sum

    mu, cov = env.get_covariance_and_return(window_size=window_size)
    return mu, cov, weights


# ===============================
# Meta Portfolio Environment (Improved) with allocation logging
# ===============================
class MetaPortfolioEnv(gym.Env):
    def __init__(self, mu, cov, amount, risk, duration,
                 micro_indicators, macro_indicators, regimes, num_regimes=3):
        super(MetaPortfolioEnv, self).__init__()
        self.mu = mu
        self.cov = cov
        self.amount = amount
        self.risk = risk
        self.duration = duration
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.regimes = regimes
        self.num_regimes = num_regimes

        self.n_assets = len(mu)
        self.micro_dim = micro_indicators.shape[1]
        self.macro_dim = macro_indicators.shape[1]

        obs_dim = self.n_assets * 3 + 1 + self.micro_dim + self.macro_dim + self.num_regimes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.current_step = 0
        self.portfolio_value = amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.allocation_history = []

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.allocation_history = []
        return self._get_obs()

    def _get_obs(self):
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
        self.current_step += 1
        weights = np.clip(action, 0, 1)
        total = np.sum(weights)
        if total < 1e-8:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total

        expected_return = np.dot(weights, self.mu)
        variance = np.dot(weights.T, np.dot(self.cov, weights))
        risk_penalty = self.risk * variance

        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + expected_return)

        # Log allocation = weights * portfolio_value at this step
        allocation = weights * self.portfolio_value
        self.allocation_history.append(allocation)

        log_return = np.log(self.portfolio_value / (prev_value + 1e-12) + 1e-12)
        reward = log_return - risk_penalty

        done = self.current_step >= self.duration
        self.prev_weights = weights

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": log_return,
            "risk_penalty": risk_penalty,
            "reward": reward
        }
        return self._get_obs(), reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


# ===============================
# Helper to get last prices
# ===============================
def _get_last_prices_for_tickers(prices_stock, prices_crypto, prices_etf, tickers_stock, tickers_crypto, tickers_etf):
    last_stock = prices_stock[-1] if (prices_stock is not None and len(prices_stock) > 0) else np.array([])
    last_crypto = prices_crypto[-1] if (prices_crypto is not None and len(prices_crypto) > 0) else np.array([])
    last_etf = prices_etf[-1] if (prices_etf is not None and len(prices_etf) > 0) else np.array([])
    last_prices = np.concatenate([last_stock, last_crypto, last_etf]) if (len(last_stock) + len(last_crypto) + len(last_etf)) > 0 else np.array([])
    all_tickers = tickers_stock + tickers_crypto + tickers_etf
    if len(all_tickers) != len(last_prices):
        # fallback to ones to avoid crash, but we inform user later
        last_prices = np.ones(len(all_tickers))
    return all_tickers, last_prices


# ===============================
# Streamlit App Main
# ===============================
st.set_page_config(page_title="Meta Portfolio + Alpaca Paper Trading", layout="wide")
st.title("Meta Portfolio Allocation with SAC Meta-Agent + Alpaca Paper Trading")

# Sidebar: paper trading controls (persistent)
st.sidebar.header("Paper Trading (Alpaca)")

# Checkbox toggles enable state and persists it
enable_checkbox = st.sidebar.checkbox("Enable Paper Trading", value=st.session_state["paper_trading_enabled"])
st.session_state["paper_trading_enabled"] = enable_checkbox

if st.session_state["paper_trading_enabled"]:
    st.sidebar.markdown("**Alpaca Paper Account Keys** (kept in session only)")
    st.session_state["alpaca_api_key"] = st.sidebar.text_input("API Key", value=st.session_state["alpaca_api_key"])
    st.session_state["alpaca_secret_key"] = st.sidebar.text_input("Secret Key", value=st.session_state["alpaca_secret_key"], type="password")
    st.session_state["alpaca_base_url"] = st.sidebar.text_input("Base URL", value=st.session_state["alpaca_base_url"])

    if st.sidebar.button("Save & Connect to Alpaca"):
        if tradeapi is None:
            st.sidebar.error("alpaca-trade-api package not installed. Run `pip install alpaca-trade-api`.")
        elif (not st.session_state["alpaca_api_key"]) or (not st.session_state["alpaca_secret_key"]):
            st.sidebar.error("Please fill both API Key and Secret.")
        else:
            try:
                client = tradeapi.REST(
                    st.session_state["alpaca_api_key"],
                    st.session_state["alpaca_secret_key"],
                    base_url=st.session_state["alpaca_base_url"],
                    api_version='v2'
                )
                account = client.get_account()
                st.sidebar.success(f"Connected to Alpaca (status: {account.status})")
                st.session_state["alpaca_client"] = client
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {e}")
                st.session_state["alpaca_client"] = None
else:
    # If disabled, clear client
    st.session_state["alpaca_client"] = None

# Load model/data (cached)
@st.cache_resource(show_spinner=False)
def load_all_data():
    stock_data = load_agent_data("stock")
    crypto_data = load_agent_data("crypto")
    etf_data = load_agent_data("etf")
    return stock_data, crypto_data, etf_data

# Main UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulation Inputs")
    try:
        stock_data, crypto_data, etf_data = load_all_data()
        tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock, scaler_stock, macro_scaler_stock = stock_data
        tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto, scaler_crypto, macro_scaler_crypto = crypto_data
        tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf, scaler_etf, macro_scaler_etf = etf_data
        load_error = None
    except Exception as e:
        tickers_stock = tickers_crypto = tickers_etf = []
        prices_stock = prices_crypto = prices_etf = np.array([])
        regimes_stock = regimes_crypto = regimes_etf = np.array([])
        micro_stock = micro_crypto = micro_etf = np.array([[]])
        macro_stock = macro_crypto = macro_etf = np.array([[]])
        ppo_stock = ppo_crypto = ppo_etf = None
        st.error(f"Error loading models/data: {e}")
        load_error = e

    amount = st.number_input("Total Investment Amount ($)", min_value=1000.0, max_value=10_000_000.0, value=10000.0, step=1000.0, format="%.2f")
    risk = st.slider("Risk Appetite (0: low - 1: high)", 0.0, 1.0, 0.5, 0.01)
    duration = st.number_input("Investment Duration (days)", min_value=1, max_value=3650, value=30, step=1)

    run_button = st.button("Run Portfolio Allocation")

with col2:
    st.subheader("Simulation Output")
    output_placeholder = st.empty()

# Run allocation and simulation
if run_button:
    if load_error:
        st.error("Cannot run allocation due to load error. Fix model/data loading first.")
    else:
        # Create agent envs
        env_stock = AgentEnv(prices_stock, regimes_stock, micro_stock, macro_stock)
        env_crypto = AgentEnv(prices_crypto, regimes_crypto, micro_crypto, macro_crypto)
        env_etf = AgentEnv(prices_etf, regimes_etf, micro_etf, macro_etf)

        mu_stock, cov_stock, w_stock = get_agent_portfolio(ppo_stock, env_stock)
        mu_crypto, cov_crypto, w_crypto = get_agent_portfolio(ppo_crypto, env_crypto)
        mu_etf, cov_etf, w_etf = get_agent_portfolio(ppo_etf, env_etf)

        mu_all = np.concatenate([mu_stock, mu_crypto, mu_etf])
        cov_all = np.block([
            [cov_stock, np.zeros((len(mu_stock), len(mu_crypto))), np.zeros((len(mu_stock), len(mu_etf)))],
            [np.zeros((len(mu_crypto), len(mu_stock))), cov_crypto, np.zeros((len(mu_crypto), len(mu_etf)))],
            [np.zeros((len(mu_etf), len(mu_stock))), np.zeros((len(mu_etf), len(mu_crypto))), cov_etf]
        ])

        min_rows = min(
            micro_stock.shape[0] if hasattr(micro_stock, "shape") else 0,
            micro_crypto.shape[0] if hasattr(micro_crypto, "shape") else 0,
            micro_etf.shape[0] if hasattr(micro_etf, "shape") else 0,
            macro_stock.shape[0] if hasattr(macro_stock, "shape") else 0,
            macro_crypto.shape[0] if hasattr(macro_crypto, "shape") else 0,
            macro_etf.shape[0] if hasattr(macro_etf, "shape") else 0
        )
        if min_rows <= 0:
            st.error("Indicator arrays have invalid shapes. Ensure micro/macro arrays have rows.")
        else:
            combined_micro = np.hstack([
                micro_stock[:min_rows],
                micro_crypto[:min_rows],
                micro_etf[:min_rows]
            ])

            combined_macro = np.hstack([
                macro_stock[:min_rows],
                macro_crypto[:min_rows],
                macro_etf[:min_rows]
            ])

            combined_regimes_result = mode(np.vstack([
                regimes_stock[:min_rows], regimes_crypto[:min_rows], regimes_etf[:min_rows]
            ]), axis=0)
            combined_regimes = combined_regimes_result.mode.squeeze()

            meta_env = MetaPortfolioEnv(
                mu=mu_all,
                cov=cov_all,
                amount=amount,
                risk=risk,
                duration=duration,
                micro_indicators=combined_micro,
                macro_indicators=combined_macro,
                regimes=combined_regimes
            )
            meta_env.seed(42)

            # Load SAC meta model
            try:
                meta_model = SAC.load("sac_meta_model")
            except Exception as e:
                st.error(f"Failed to load SAC meta model: {e}")
                meta_model = None

            if meta_model is None:
                st.warning("Meta model not loaded; cannot simulate allocations without 'sac_meta_model'.")
            else:
                obs = meta_env.reset()
                allocation_log = []
                done = False
                while not done:
                    obs_reshaped = obs.reshape(1, -1)
                    action, _ = meta_model.predict(obs_reshaped, deterministic=True)
                    weights = np.clip(action.flatten(), 0, 1)
                    weights_sum = weights.sum()
                    if weights_sum < 1e-8:
                        weights = np.ones_like(weights) / len(weights)
                    else:
                        weights /= weights_sum

                    # enforce minimum stock allocation 50%
                    stock_len = len(tickers_stock)
                    crypto_len = len(tickers_crypto)
                    etf_len = len(tickers_etf)

                    stock_indices = np.arange(stock_len)
                    crypto_indices = np.arange(stock_len, stock_len + crypto_len)
                    etf_indices = np.arange(stock_len + crypto_len, stock_len + crypto_len + etf_len)

                    stock_weight = weights[stock_indices].sum() if stock_len > 0 else 0.0
                    crypto_weight = weights[crypto_indices].sum() if crypto_len > 0 else 0.0
                    etf_weight = weights[etf_indices].sum() if etf_len > 0 else 0.0

                    min_stock_weight = 0.5
                    if stock_weight < min_stock_weight and stock_len > 0:
                        diff = min_stock_weight - stock_weight
                        total_other = crypto_weight + etf_weight
                        if total_other > 0:
                            scale = max((total_other - diff) / (total_other + 1e-12), 0.0)
                            if crypto_len > 0:
                                weights[crypto_indices] *= scale
                            if etf_len > 0:
                                weights[etf_indices] *= scale
                        else:
                            if stock_len > 0:
                                weights[stock_indices] = 1.0
                            if crypto_len > 0:
                                weights[crypto_indices] = 0.0
                            if etf_len > 0:
                                weights[etf_indices] = 0.0

                        new_stock_weight = weights[stock_indices].sum() if stock_len > 0 else 0.0
                        if new_stock_weight > 0:
                            weights[stock_indices] *= (stock_weight + diff) / (new_stock_weight + 1e-12)
                        weights /= weights.sum()

                    obs, reward, done, info = meta_env.step(weights)
                    allocation_log.append(info["weights"] * meta_env.portfolio_value)

                # Save log & tickers to session
                st.session_state["allocation_log"] = allocation_log
                st.session_state["all_tickers"] = tickers_stock + tickers_crypto + tickers_etf

                # Produce summary table & plots
                portfolio_values = [amount]
                for alloc in allocation_log:
                    denom = portfolio_values[-1] if portfolio_values[-1] != 0 else 1e-12
                    daily_return = np.dot(alloc / denom, mu_all)
                    portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
                profit = portfolio_values[-1] - amount

                # summary
                summary_df = pd.DataFrame({
                    "Metric": ["Total Investment Amount", f"Projected Valuation (after {duration} days)", "Projected Profit"],
                    "Value": [f"${amount:,.2f}", f"${portfolio_values[-1]:,.2f}", f"${profit:,.2f}"]
                })
                output_placeholder.subheader("Meta Portfolio Allocation Summary")
                output_placeholder.table(summary_df)

                # allocations DataFrame
                final_alloc = allocation_log[-1]
                all_tickers = st.session_state["all_tickers"]
                alloc_df = pd.DataFrame({
                    "Asset": all_tickers,
                    "Allocation ($)": final_alloc,
                    "Allocation (%)": final_alloc / (final_alloc.sum() + 1e-12) * 100
                })
                alloc_df_display = alloc_df.copy()
                alloc_df_display["Allocation ($)"] = alloc_df_display["Allocation ($)"].map("${:,.2f}".format)
                alloc_df_display["Allocation (%)"] = alloc_df_display["Allocation (%)"].map("{:.2f}%".format)
                output_placeholder.subheader("Asset Allocations on Final Day")
                output_placeholder.dataframe(alloc_df_display)

                # plot portfolio values
                with plt.style.context('ggplot'):
                    fig_val, ax_val = plt.subplots()
                    ax_val.plot(range(len(portfolio_values)), portfolio_values, linewidth=2)
                    ax_val.set_xlabel("Days")
                    ax_val.set_ylabel("Portfolio Value ($)")
                    ax_val.set_title("Projected Portfolio Growth")
                    ax_val.grid(True)
                    output_placeholder.pyplot(fig_val)

                # pie charts
                output_placeholder.subheader("Asset Allocations by Category")
                def plot_pie_chart(ax, labels, sizes, title):
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
                    centre_circle = plt.Circle((0,0),0.70,fc='white')
                    ax.add_artist(centre_circle)
                    ax.axis('equal')
                    ax.set_title(title)

                cols = output_placeholder.columns(3)
                idx = 0
                for col, (agent_name, tickers_list) in zip(cols, [("Stock", tickers_stock), ("Crypto", tickers_crypto), ("ETF", tickers_etf)]):
                    labels = tickers_list
                    sizes = final_alloc[idx: idx + len(tickers_list)]
                    idx += len(tickers_list)
                    total_alloc = sum(sizes)
                    sizes_pct = [(s / (total_alloc + 1e-12) * 100) if total_alloc > 0 else 0 for s in sizes]
                    fig, ax = plt.subplots()
                    plot_pie_chart(ax, labels, sizes_pct, f"{agent_name} Allocations (%)")
                    col.pyplot(fig)

# ---------- Show daily allocation log button ----------
if st.button("Show Daily Allocations Log"):
    if st.session_state["allocation_log"] is not None:
        all_tickers = st.session_state.get("all_tickers", [f"Asset {i}" for i in range(len(st.session_state["allocation_log"][0]))])
        alloc_log_df = pd.DataFrame(st.session_state["allocation_log"], columns=all_tickers)
        st.dataframe(alloc_log_df)
    else:
        st.info("No allocation log available. Run portfolio allocation first.")

# ---------- Paper trade submission ----------
st.sidebar.markdown("---")
st.sidebar.header("Submit Final Allocation to Alpaca (Paper)")

if st.sidebar.button("Prepare Submit Orders"):
    if st.session_state["allocation_log"] is None:
        st.sidebar.error("No allocation_log found. Run allocation first.")
    else:
        st.sidebar.info("Ready to submit orders. Configure options below.")

# Order options
frac_allowed = st.sidebar.checkbox("Allow fractional shares", value=False)
submit_confirm = st.sidebar.button("Submit Paper Orders (final allocation)")



if submit_confirm:
    if st.session_state["allocation_log"] is None:
        st.sidebar.error("No allocation_log found. Run portfolio allocation first.")
    elif not st.session_state["paper_trading_enabled"]:
        st.sidebar.error("Paper trading disabled. Enable it in the sidebar.")
    elif tradeapi is None:
        st.sidebar.error("alpaca-trade-api package not installed. Run `pip install alpaca-trade-api`.")
    elif (not st.session_state["alpaca_api_key"]) or (not st.session_state["alpaca_secret_key"]):
        st.sidebar.error("API keys not set. Enter keys in the sidebar and Save & Connect.")
    elif st.session_state["alpaca_client"] is None:
        st.sidebar.error("Not connected to Alpaca. Click 'Save & Connect to Alpaca' in the sidebar.")
    else:
        client = st.session_state["alpaca_client"]
        final_alloc = np.array(st.session_state["allocation_log"][-1])
        all_tickers, last_prices = _get_last_prices_for_tickers(prices_stock, prices_crypto, prices_etf,
                                                               tickers_stock, tickers_crypto, tickers_etf)

        orders_report = []
        for symbol, dollars, price in zip(all_tickers, final_alloc, last_prices):
            if price is None or price == 0 or np.isnan(price):
                orders_report.append((symbol, "skipped", "no price"))
                continue

            if frac_allowed:
                qty = dollars / price
            else:
                qty = math.floor(dollars / price)

            if qty <= 0:
                orders_report.append((symbol, "skipped", "qty 0"))
                continue

            try:
                order = client.submit_order(
                    symbol=symbol,
                    qty=float(qty) if frac_allowed else int(qty),
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                orders_report.append((symbol, "submitted", f"qty={qty}, id={order.id}"))
            except Exception as e:
                orders_report.append((symbol, "error", str(e)))

        report_df = pd.DataFrame(orders_report, columns=["symbol", "status", "details"])
        st.sidebar.subheader("Order Submission Report")
        st.sidebar.dataframe(report_df)

