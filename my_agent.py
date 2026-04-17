"""CryptoScout — a Strands agent that produces rule-based crypto trading signals.

Data sources (no API key required):
  - CoinGecko /simple/price  -> current spot price & 24h change
  - Binance  /api/v3/klines  -> OHLCV candles for indicator computation

Indicators (computed in-process with pandas):
  - RSI(14)          - Wilder's smoothing via EWM(alpha=1/length)
  - MACD(12, 26, 9)  - EMA-based momentum + signal line + histogram
  - SMA(50) / SMA(200) - classic trend filter ("golden/death cross")

Requires a Gemini API key in the GEMINI_API_KEY (or GOOGLE_API_KEY) env var.
Educational use only. Not financial advice.
"""

from __future__ import annotations

import pandas as pd
import requests
from strands import Agent, tool
from strands.models.gemini import GeminiModel
from strands_tools import calculator, http_request

# --- Scoring thresholds (tune these to change signal sensitivity) -----------

RSI_OVERSOLD = 30       # RSI below this contributes +1 to the score (bullish)
RSI_OVERBOUGHT = 70     # RSI above this contributes -1 (bearish)
BUY_THRESHOLD = 2       # score >=  BUY_THRESHOLD  -> BUY
SELL_THRESHOLD = -2     # score <= SELL_THRESHOLD  -> SELL


# --- Indicator helpers ------------------------------------------------------

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram (all aligned to `close`'s index)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


# --- Custom tools -----------------------------------------------------------

@tool
def get_price(coingecko_id: str) -> dict:
    """Get current USD spot price and 24h change for a crypto asset.

    Args:
        coingecko_id: CoinGecko asset id, e.g. 'bitcoin', 'ethereum', 'solana'.
    """
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": coingecko_id.lower(),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@tool
def get_ohlcv(binance_symbol: str, interval: str = "1h", limit: int = 300) -> list:
    """Fetch recent OHLCV candles from Binance's public API.

    Args:
        binance_symbol: Binance pair, e.g. 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'.
        interval: One of '1m','5m','15m','1h','4h','1d'. Default '1h'.
        limit: Number of candles (max 1000). Default 300.

    Returns: list of [open_time_ms, open, high, low, close, volume].
    """
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": binance_symbol.upper(), "interval": interval, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    return [
        [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        for row in r.json()
    ]


@tool
def technical_indicators(binance_symbol: str, interval: str = "1h") -> dict:
    """Compute RSI(14), MACD(12,26,9), SMA(50) and SMA(200) on recent candles
    and emit a simple rule-based BUY/SELL/HOLD signal.

    Args:
        binance_symbol: Binance pair, e.g. 'BTCUSDT'.
        interval: Candle timeframe (default '1h'). Use '4h' or '1d' for swing signals.
    """
    # Pull 300 candles so SMA(200) has enough warm-up room
    candles = get_ohlcv(binance_symbol, interval, 300)
    df = pd.DataFrame(candles, columns=["t", "o", "h", "l", "c", "v"])
    close = df["c"]

    df["rsi"] = _rsi(close, 14)
    df["macd"], df["macd_sig"], df["macd_hist"] = _macd(close)
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()

    # Evaluate the three rules on the most recent closed candle
    last = df.iloc[-1]
    reasons: list[str] = []
    score = 0

    if last["rsi"] < RSI_OVERSOLD:
        score += 1
        reasons.append(f"RSI {last['rsi']:.1f} oversold (<{RSI_OVERSOLD})")
    elif last["rsi"] > RSI_OVERBOUGHT:
        score -= 1
        reasons.append(f"RSI {last['rsi']:.1f} overbought (>{RSI_OVERBOUGHT})")
    else:
        reasons.append(f"RSI {last['rsi']:.1f} neutral")

    if last["macd"] > last["macd_sig"]:
        score += 1
        reasons.append("MACD above signal (bullish momentum)")
    else:
        score -= 1
        reasons.append("MACD below signal (bearish momentum)")

    if last["sma50"] > last["sma200"]:
        score += 1
        reasons.append("SMA50 > SMA200 (bullish trend)")
    else:
        score -= 1
        reasons.append("SMA50 < SMA200 (bearish trend)")

    if score >= BUY_THRESHOLD:
        signal = "BUY"
    elif score <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": binance_symbol.upper(),
        "interval": interval,
        "price": round(float(last["c"]), 4),
        "rsi": round(float(last["rsi"]), 2),
        "macd": round(float(last["macd"]), 4),
        "macd_signal": round(float(last["macd_sig"]), 4),
        "sma50": round(float(last["sma50"]), 4),
        "sma200": round(float(last["sma200"]), 4),
        "rule_based_signal": signal,
        "score": score,
        "reasons": reasons,
    }


# --- Agent ------------------------------------------------------------------

SYSTEM_PROMPT = """You are CryptoScout, a crypto market analyst.

When the user asks for a trading signal for a coin:
1. Call `technical_indicators` with the correct Binance pair (e.g. BTCUSDT,
   ETHUSDT, SOLUSDT). Default to the '1h' interval unless the user says
   otherwise; offer '4h' or '1d' for swing trades.
2. Optionally call `get_price` with the CoinGecko id (e.g. 'bitcoin') for a
   quick spot check or 24h change.
3. Combine the indicators into ONE recommendation: BUY, SELL, or HOLD.
4. Explain in plain English, citing the actual numbers returned.
5. Always end with the exact disclaimer:
   "⚠️ Educational output only. Not financial advice. Crypto is volatile — DYOR."

Rules:
- Never claim to predict the future.
- If a tool errors or data is missing, say so clearly instead of guessing.
- Prefer concise, structured answers (bulleted reasons, one-line verdict).
"""

model = GeminiModel(
    model_id="gemini-2.5-flash",
    params={"max_output_tokens": 1024},
)

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_price, get_ohlcv, technical_indicators, calculator, http_request],
)


if __name__ == "__main__":
    print("CryptoScout is ready. Try: 'Give me a 4h signal for BTCUSDT'.")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue
        response = agent(user_input)
        print(f"Scout: {response}\n")
