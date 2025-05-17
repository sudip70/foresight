# 🧠 Stock Portfolio Allocation & Risk Management using Reinforcement Learning + LLM Sentiment

This project combines **Reinforcement Learning (RL)**, **contextual bandits**, and **large language models (LLMs)** to create a multi-agent financial decision-making system for real-time **stock portfolio allocation and risk management**.

Agents specialize in different asset classes (e.g., stocks, ETFs, options, crypto) and merge their strategies via **meta-RL**. The system is **regime-aware** (bull/bear/sideways) and uses **macroeconomic indicators** (inflation, GDP, interest rates) for context-aware policy selection. Sentiment signals are extracted from social media using LLMs and integrated into the trading pipeline.

---

## 🔧 Features

* **Multi-Agent RL System**:

  * Specialized agents for equities, ETFs, options, and crypto.
  * Merged using **meta-reinforcement learning** and **contextual bandits**.

* **Market Regime Awareness**:

  * Detects bull, bear, and sideways markets.
  * Factors in macroeconomic conditions (e.g., inflation, recession indicators).

* **LLM-Powered Sentiment Analysis**:

  * Real-time social media (Twitter/X, Reddit, financial news).
  * Uses fine-tuned or zero-shot LLMs for market sentiment extraction.
  * Sentiment integrated as contextual signals for agents.

* **Explainability & Interpretability**:

  * SHAP, LIME, and attention visualizations to interpret model decisions.
  * Agent-level and asset-level decision explanations.

* **Real-Time Paper Trading**:

  * Hooks into **Alpaca**, **Interactive Brokers (IBKR) Sandbox**, or **QuantConnect**.
  * Executes trades and adjusts portfolio live via APIs.
  * Tracks performance metrics and adapts policies on-the-fly.

* **Risk Management**:

  * Dynamic portfolio rebalancing based on risk scores.
  * Uses Value at Risk (VaR), CVaR, and volatility-adjusted returns.
  * Handles drawdowns, exposure limits, and stop-loss triggers.

---

## 🏐 Architecture

```
[Data Ingestion] --> [Sentiment Analysis (LLM)] --->|
[Market Regime Detection]                          |
[Macroeconomic Indicators] ---------------------> [Contextual RL Agents]
                                                   ↓
                                   [Meta-RL Policy Merger / Bandit Selector]
                                                   ↓
                                    [Portfolio Allocation + Risk Manager]
                                                   ↓
                                      [Paper Trading API Integration]
                                                   ↓
                                          [Explainability Module]
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rl-stock-allocation.git
cd rl-stock-allocation
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file and add credentials for:

* Alpaca / Interactive Brokers API
* OpenAI or HuggingFace (for LLMs)
* Reddit/Twitter API (for sentiment)

```env
ALPACA_API_KEY=your_key
ALPACA_SECRET=your_secret
OPENAI_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_token
```

---

## 📊 Datasets Used

* Historical price data (Yahoo Finance, Alpaca, Quandl)
* Macroeconomic indicators (FRED API)
* Reddit & Twitter sentiment (Pushshift, Tweepy)
* News headlines (NewsAPI, Finnhub)

---

## 🧠 Models and Algorithms

* **RL Algorithms**: PPO, A2C, DDPG (via `stable-baselines3`)
* **Contextual Bandits**: LinUCB, Thompson Sampling
* **Meta-RL**: PEARL, MAML-style merging of agents
* **LLMs**: OpenAI GPT, Falcon, LLaMA2 (with financial fine-tuning)
* **Explainability**: SHAP, attention heatmaps

---

## 📊 Evaluation Metrics

* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Alpha/Beta
* Policy Improvement over baseline
* Trade accuracy with sentiment

---

## 🔐 API Integrations

* 🟢 **Alpaca** ([https://alpaca.markets](https://alpaca.markets))
* 🔣 **Interactive Brokers** (sandbox mode)
* 🟣 **OpenAI / HuggingFace**
* 🔹 **Twitter**, **Reddit**, **FRED**, **NewsAPI**

---

## 🧪 Project Status

* [x] Initial agents trained for each asset type
* [x] LLM-based sentiment integrated
* [x] Market regime detector working (trend/macro)
* [x] Real-time paper trading setup via Alpaca
* [ ] Performance dashboard and live analytics
* [ ] Backtest results visualization

---

## 📂 Folder Structure

```
📁 agents/                 # RL agents for each asset class
📁 data/                   # Datasets and preprocessing scripts
📁 envs/                   # Custom OpenAI Gym environments
📁 sentiment/              # LLM and NLP-based sentiment module
📁 trading/                # Real-time trading hooks (Alpaca, IBKR)
📁 explainability/         # SHAP, LIME, and visual interpreters
📁 config/                 # Config files and hyperparameters
📁 notebooks/              # EDA and experimentation notebooks
📁 utils/                  # Utility scripts
🔍 main.py                 # Entry point
🔍 requirements.txt
```

---

## 🤖 Future Roadmap

* ✅ Add performance dashboard using Streamlit or Dash
* ↻ Support reinforcement learning fine-tuning with live feedback
* 🦹‍♂️ Expand to crypto-specific signals (on-chain data, sentiment)
* 📉 Add adversarial agents for stress testing portfolio strategies
* 🧠 Plug-in LLMs to generate investment rationales for trades

---

## 🙋 Contributing

PRs are welcome! Please open an issue first to discuss major changes.

---

## 📜 License

MIT License — feel free to use, modify, and build upon this.

---

## 🔗 References

* [Alpaca Docs](https://alpaca.markets/docs/)
* [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [OpenAI API](https://platform.openai.com/docs)
* [Market Sentiment LLM papers](https://arxiv.org)

---

## ✨ Acknowledgements

Thanks to open-source contributors, financial research datasets, and the ML research community for inspiration.
