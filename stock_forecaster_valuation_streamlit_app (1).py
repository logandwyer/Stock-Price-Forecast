# app.py
# -------------------------------------------------------------
# Stock Forecaster & Valuation Dashboard (Streamlit)
# One input: ticker (e.g., AAPL)
# Pulls public market/company data via yfinance, generates
# 1D, 5D, 1M, 6M, 1Y forecasts using:
#   • GBM/lognormal (analytical)
#   • ARIMA (statsmodels) time‑series model
# Also:
#   • Monte Carlo paths & probability of hitting targets
#   • Company snapshot (trend, margins, growth, ROE, leverage)
#   • Heuristic Over/Under/Fair value + optional lightweight DCF
# -------------------------------------------------------------

import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import norm

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# ---------------------- Constants ----------------------
TRADING_DAYS = 252
HORIZONS = {
    "1 Day": 1/252,
    "5 Days": 5/252,
    "1 Month (~21d)": 21/252,
    "6 Months (~126d)": 126/252,
    "1 Year (252d)": 252/252,
}

# ---------------------- Caching ----------------------
@st.cache_data(show_spinner=False)
def load_history(ticker: str, years: int = 5) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 10)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Could not load price history. Check the ticker.")
    df = df.dropna()
    return df

@st.cache_data(show_spinner=False)
def load_info(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        fi = tk.fast_info
        info.update({
            'lastPrice': float(fi.last_price) if getattr(fi, 'last_price', None) is not None else None,
            'marketCap': float(fi.market_cap) if getattr(fi, 'market_cap', None) is not None else None,
            'sharesOutstanding': float(fi.shares_outstanding) if getattr(fi, 'shares_outstanding', None) is not None else None,
        })
    except Exception:
        pass
    try:
        raw = tk.info
        if isinstance(raw, dict):
            for k in [
                'shortName','longName','industry','sector','country','website',
                'trailingPE','forwardPE','priceToBook','priceToSalesTrailing12Months',
                'dividendYield','beta','profitMargins','grossMargins','operatingMargins',
                'debtToEquity','returnOnEquity','revenueGrowth','earningsGrowth','currentRatio',
                'longBusinessSummary'
            ]:
                info[k] = raw.get(k, None)
    except Exception:
        pass
    try:
        cf = tk.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            fcf_candidates = ['Free Cash Flow', 'FreeCashFlow', 'freeCashFlow']
            found = None
            for c in cf.index:
                if str(c).lower().replace(' ', '') in [x.lower().replace(' ', '') for x in fcf_candidates]:
                    found = c
                    break
            if found is not None:
                info['freeCashFlow'] = float(cf.loc[found].dropna().iloc[0])
    except Exception:
        pass
    return info

# ---------------------- Math helpers ----------------------

def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    if prices.index.max() - prices.index.min() < pd.Timedelta(days=lookback_years*365 - 10):
        px = prices.copy()
    else:
        cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
        px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily = float(rets.mean())
    sigma_daily = float(rets.std())
    mu_annual = mu_daily * TRADING_DAYS
    sigma_annual = sigma_daily * math.sqrt(TRADING_DAYS)
    return mu_annual, sigma_annual


def gbm_distribution(s0: float, mu: float, sigma: float, t_years: float) -> Tuple[float, float]:
    m = math.log(s0) + (mu - 0.5*sigma**2) * t_years
    v = (sigma**2) * t_years
    return m, v


def gbm_points(s0: float, mu: float, sigma: float, t_years: float) -> Dict[str, float]:
    m, v = gbm_distribution(s0, mu, sigma, t_years)
    sd = math.sqrt(v)
    median = math.exp(m)
    expected = s0 * math.exp(mu * t_years)
    p05 = math.exp(m - 1.9599639845*sd)
    p95 = math.exp(m + 1.9599639845*sd)
    return {"expected": expected, "median": median, "p05": p05, "p95": p95}


def gbm_prob_reach(s0: float, mu: float, sigma: float, t_years: float, target: float) -> float:
    if target <= 0:
        return 1.0
    m, v = gbm_distribution(s0, mu, sigma, t_years)
    z = (math.log(target) - m) / math.sqrt(v)
    return float(1 - norm.cdf(z))


def simulate_paths(s0: float, mu: float, sigma: float, days: int, n_paths: int = 2000, seed: Optional[int] = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal(0, 1, size=(days, n_paths))
    paths = np.zeros((days+1, n_paths))
    paths[0, :] = s0
    drift = (mu - 0.5*sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    for t in range(1, days+1):
        paths[t, :] = paths[t-1, :] * np.exp(drift + vol * shocks[t-1, :])
    return paths


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, window//5)).mean()

# ---------------------- Valuation ----------------------

def heuristic_valuation_flags(info: Dict, prices: pd.Series) -> Tuple[str, Dict[str, str]]:
    reasons = {}
    votes = []
    pe = info.get('trailingPE', None)
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 12: votes.append(-1); reasons['PE'] = f"Trailing P/E {pe:.1f} (low)"
        elif pe > 30: votes.append(1); reasons['PE'] = f"Trailing P/E {pe:.1f} (high)"
        else: votes.append(0); reasons['PE'] = f"Trailing P/E {pe:.1f} (mid)"
    ps = info.get('priceToSalesTrailing12Months', None)
    if isinstance(ps, (int, float)) and ps > 0:
        if ps < 2: votes.append(-1); reasons['P/S'] = f"P/S {ps:.1f} (low)"
        elif ps > 10: votes.append(1); reasons['P/S'] = f"P/S {ps:.1f} (very high)"
        else: votes.append(0); reasons['P/S'] = f"P/S {ps:.1f} (mid)"
    pb = info.get('priceToBook', None)
    if isinstance(pb, (int, float)) and pb > 0:
        if pb < 1: votes.append(-1); reasons['P/B'] = f"P/B {pb:.1f} (below book)"
        elif pb > 6: votes.append(1); reasons['P/B'] = f"P/B {pb:.1f} (high)"
        else: votes.append(0); reasons['P/B'] = f"P/B {pb:.1f} (mid)"
    last = float(prices.iloc[-1])
    ma200 = float(sma(prices, 200).iloc[-1]) if len(prices) >= 200 else None
    if ma200 and ma200 > 0:
        dev = (last/ma200 - 1.0) * 100
        reasons['Trend'] = f"Price is {dev:+.1f}% vs 200‑day MA"
        if dev < -10: votes.append(-0.5)
        elif dev > +20: votes.append(0.5)
        else: votes.append(0)
    if not votes:
        return ("Inconclusive (insufficient data)", reasons)
    score = np.mean(votes)
    if score <= -0.25:
        verdict = "Likely Undervalued (heuristic)"
    elif score >= 0.25:
        verdict = "Likely Overvalued (heuristic)"
    else:
        verdict = "Around Fair Value (heuristic)"
    return verdict, reasons


def lightweight_dcf(info: Dict, price: float) -> Optional[Dict[str, float]]:
    fcf = info.get('freeCashFlow', None)
    shares = info.get('sharesOutstanding', None)
    beta = info.get('beta', 1.0) or 1.0
    if not (isinstance(fcf, (int, float)) and isinstance(shares, (int, float)) and shares > 0):
        return None
    rf = st.session_state.get('rf_rate', 0.04)
    mrp = st.session_state.get('mkt_premium', 0.055)
    g = st.session_state.get('terminal_g', 0.02)
    ke = rf + beta * mrp
    if ke <= g:
        g = ke - 0.005
    fcf_next = fcf * (1 + g)
    equity_value = fcf_next / (ke - g)
    per_share = equity_value / shares
    disc = {
        'rf': rf, 'mrp': mrp, 'beta': beta, 'ke': ke, 'g': g,
        'fcf_ttm': fcf, 'shares': shares, 'intrinsic': per_share,
        'premium_vs_price_%': (per_share/price - 1) * 100
    }
    return disc

# ---------------------- Company Blurb ----------------------

def company_position_blurb(info: Dict, prices: pd.Series) -> str:
    pieces = []
    name = info.get('longName') or info.get('shortName') or ''
    sector = info.get('sector') or ''
    industry = info.get('industry') or ''
    last = float(prices.iloc[-1])
    ma50 = float(sma(prices, 50).iloc[-1]) if len(prices) >= 50 else None
    ma200 = float(sma(prices, 200).iloc[-1]) if len(prices) >= 200 else None
    if ma50:
        pieces.append(f"Price ${last:,.2f} vs 50‑day MA ${ma50:,.2f}")
    if ma200:
        pieces.append(f"200‑day MA ${ma200:,.2f}")
    rev_g = info.get('revenueGrowth')
    pm = info.get('profitMargins')
    roe = info.get('returnOnEquity')
    dte = info.get('debtToEquity')
    if isinstance(rev_g, (int, float)):
        pieces.append(f"TTM revenue growth {rev_g*100:.1f}%")
    if isinstance(pm, (int, float)):
        pieces.append(f"Profit margin {pm*100:.1f}%")
    if isinstance(roe, (int, float)):
        pieces.append(f"ROE {roe*100:.1f}%")
    if isinstance(dte, (int, float)):
        pieces.append(f"Debt/Equity {dte:.1f}")
    profile_line = ", ".join([p for p in [name, sector, industry] if p])
    summary = info.get('longBusinessSummary')
    blurb = f"**{profile_line}**. " if profile_line else ""
    if pieces:
        blurb += " | ".join(pieces) + ". "
    if summary:
        blurb += f"\n\n**Company Summary:** {summary}"
    return blurb

# -------------------------------------------------------------
# (Rest of the code remains same as previous version with backtest, comps, and export sections)
