# CryptoScout

A small AI agent that produces **rule-based crypto trading signals**. It pulls live market data, computes standard technical indicators, combines them into a BUY / SELL / HOLD verdict, and lets you chat with it in natural language powered by Google Gemini.

Built with the [Strands Agents](https://github.com/strands-agents/sdk-python) SDK.

> ⚠️ **Educational only. Not financial advice.** An LLM cannot predict prices. This project demonstrates how to give an agent tools and a deterministic rule set — nothing more.

---

## What it does

Given a user question like *"Give me a 4h signal for BTCUSDT"*, the agent:

1. Calls the `technical_indicators` tool with the requested pair and timeframe.
2. That tool fetches 300 OHLCV candles from Binance's public API and computes:
   - **RSI(14)** — Wilder's smoothing
   - **MACD(12, 26, 9)** — EMA-based momentum
   - **SMA(50)** and **SMA(200)** — trend filter
3. Scores the three rules (+1 bullish / -1 bearish each) and maps the score to **BUY** (≥ 2), **SELL** (≤ -2), or **HOLD**.
4. The LLM then explains the verdict in plain English, citing the actual numbers, and always ends with a risk disclaimer.

It can also call `get_price` for a quick CoinGecko spot price, plus the built-in `calculator` and `http_request` tools from `strands-tools`.

## Architecture

```
 User  ──►  Strands Agent (Gemini 2.5 Flash)
                │
                ▼
     ┌────────────────────────────────┐
     │ Tools                          │
     │  • get_price          (CoinGecko) │
     │  • get_ohlcv          (Binance)   │
     │  • technical_indicators (pandas)  │
     │  • calculator, http_request       │
     └────────────────────────────────┘
```

No API key is required for the data sources. You only need a **Gemini API key**.

## Requirements

- Python 3.11+ (tested on 3.13)
- A Google Gemini API key ([Get one free](https://aistudio.google.com/app/apikey))
- Internet access (CoinGecko + Binance public endpoints; some networks block Binance — CoinGecko alone still works for price)

## Setup

```bash
# 1. Clone
git clone git@github.com:Thuto42096/my-first-agent.git
cd my-first-agent

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install 'strands-agents[gemini]' strands-agents-tools pandas requests

# 4. Set your Gemini API key
export GEMINI_API_KEY="your-key-here"   # or GOOGLE_API_KEY
```

## Usage

```bash
python my_agent.py
```

Example session:

```
CryptoScout is ready. Try: 'Give me a 4h signal for BTCUSDT'.
Type 'quit' to exit.

You: give me a 1h signal for ETHUSDT
Scout: ETHUSDT (1h) — Price: $3,512.40
- RSI 58.2 neutral
- MACD above signal (bullish momentum)
- SMA50 > SMA200 (bullish trend)
Verdict: BUY (score 2/3)
⚠️ Educational output only. Not financial advice. Crypto is volatile — DYOR.

You: quit
```

Other prompts to try:

- `What's the price of solana right now?`
- `Compare BTCUSDT on the 1h and 4h timeframes`
- `Is ETH oversold on the daily?`

## How the signal is computed

The scoring is deliberately simple and transparent — see the constants at the top of `my_agent.py`:

```python
RSI_OVERSOLD = 30       # +1 to score
RSI_OVERBOUGHT = 70     # -1 to score
BUY_THRESHOLD = 2       # score >=  2  -> BUY
SELL_THRESHOLD = -2     # score <= -2  -> SELL
```

Each of the three rules (RSI, MACD cross, SMA trend) contributes ±1. Tune these constants to change sensitivity.

## Limitations

- **No predictive power.** These are lagging indicators applied to the most recent candle. No backtest has been run on this rule set.
- **No risk management.** There are no stop-loss or take-profit suggestions in the output yet.
- **Single timeframe.** The agent does not require multi-timeframe confirmation before emitting a signal.
- **No news / sentiment.** Purely technical.
- **Rate limits.** CoinGecko's free tier is ~10–30 req/min; Binance public ~1200/min. Heavy use needs caching.

## Roadmap / ideas

- [ ] Unit tests for `_rsi`, `_macd`, and the scoring function against fixtures
- [ ] `backtesting.py` harness to evaluate the rule set on historical data
- [ ] `get_news(symbol)` tool (CryptoPanic)
- [ ] ATR-based stop-loss / take-profit in the output
- [ ] Multi-timeframe confirmation (e.g. only BUY when 1h and 4h agree)
- [ ] Telegram/Discord delivery on a schedule

## License

No license specified yet — treat as all-rights-reserved until one is added.
