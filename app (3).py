# -------------------------------------------------------------
# Stock Forecaster & Valuation Dashboard (Hybrid, Real‑World Enhanced, Mean‑Only Monte Carlo)
# -------------------------------------------------------------

import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

TRADING_DAYS = 252
HORIZONS = {
    "1 Day": 1/252,
    "5 Days": 5/252,
    "1 Month (~21d)": 21/252,
    "6 Months (~126d)": 126/252,
    "1 Year (252d)": 252/252,
}

@st.cache_data(show_spinner=False)
def load_history(ticker: str, years: int = 5) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 10)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Could not load price history. Check the ticker.")
    return df.dropna()

@st.cache_data(show_spinner=False)
def load_info(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        fi = tk.fast_info
        info.update({'lastPrice': float(getattr(fi, 'last_price', np.nan)),
                     'marketCap': float(getattr(fi, 'market_cap', np.nan)),
                     'sharesOutstanding': float(getattr(fi, 'shares_outstanding', np.nan))})
    except Exception:
        pass
    try:
        raw = tk.info
        if isinstance(raw, dict):
            for k in ['shortName','longName','industry','sector','trailingPE','priceToBook','priceToSalesTrailing12Months','beta']:
                info[k] = raw.get(k)
    except Exception:
        pass
    return info

def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
    px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily, sigma_daily = float(rets.mean()), float(rets.std())
    mu_annual, sigma_annual = mu_daily*TRADING_DAYS, sigma_daily*math.sqrt(TRADING_DAYS)
    return mu_annual, sigma_annual

def simulate_paths(s0: float, mu: float, sigma: float, days: int, n_paths: int, seed: int):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal(0, 1, size=(days, n_paths))
    paths = np.zeros((days+1, n_paths))
    paths[0, :] = s0
    drift, vol = (mu - 0.5*sigma**2)*dt, sigma*math.sqrt(dt)
    for t in range(1, days+1):
        paths[t, :] = paths[t-1, :] * np.exp(drift + vol * shocks[t-1, :])
    return paths

def gbm_points(s0: float, mu: float, sigma: float, t_years: float) -> Dict[str, float]:
    m = math.log(s0) + (mu - 0.5*sigma**2)*t_years
    v = (sigma**2)*t_years
    sd = math.sqrt(v)
    expected = s0 * math.exp(mu * t_years)
    p05, p95 = math.exp(m - 1.96*sd), math.exp(m + 1.96*sd)
    return {"expected": expected, "p05": p05, "p95": p95}

def gbm_prob_reach(s0, mu, sigma, t_years, target):
    m = math.log(s0) + (mu - 0.5*sigma**2)*t_years
    v = (sigma**2)*t_years
    z = (math.log(target) - m) / math.sqrt(v)
    return float(1 - norm.cdf(z))

def compute_macro_alpha():
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=365*10)
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
        vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
        tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"]
        df = pd.DataFrame({"SPY": spy, "VIX": vix, "TNX": tnx}).dropna()
        df["TNX_dec"] = df["TNX"]/10/100
        df["mom63"] = df["SPY"]/df["SPY"].shift(63) - 1
        df["fwd21"] = df["SPY"].shift(-21)/df["SPY"] - 1
        df = df.dropna()
        X = np.column_stack([np.ones(len(df)), df["VIX"], df["TNX_dec"], df["mom63"]])
        y = df["fwd21"]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        V, T, M = df["VIX"].iloc[-1], df["TNX_dec"].iloc[-1], df["mom63"].iloc[-1]
        pred = beta[0] + beta[1]*V + beta[2]*T + beta[3]*M
        return pred*(TRADING_DAYS/21.0)
    except Exception:
        return 0.0

def valuation_alpha(info: Dict) -> float:
    adj = 0
    pe, pb, ps = info.get("trailingPE"), info.get("priceToBook"), info.get("priceToSalesTrailing12Months")
    if isinstance(pe, (int,float)) and pe>0:
        if pe<12: adj+=0.02
        elif pe>30: adj-=0.02
    if isinstance(ps, (int,float)) and ps>0:
        if ps<2: adj+=0.01
        elif ps>10: adj-=0.01
    if isinstance(pb, (int,float)) and pb>0:
        if pb<1: adj+=0.01
        elif pb>6: adj-=0.01
    return adj

st.set_page_config(page_title="Stock Forecaster (Hybrid)", layout="wide")
st.title("📈 Stock Forecaster — Hybrid (Mean‑Only Monte Carlo)")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").upper()
    lookback_years = st.slider("Return lookback (years)",1,10,3)
    use_hybrid = st.checkbox("Use hybrid (real‑world) adjustments", True)
    n_paths = st.slider("Monte Carlo paths (1Y)",500,10000,2000,step=500)
    seed = st.number_input("Random seed",min_value=0,value=42)
    target_move = st.selectbox("Target move",["+10%","+20%","+30%","-10%","-20%","-30%"],0)
    run = st.button("Run Analysis")

if run and ticker:
    prices = load_history(ticker,years=max(5,lookback_years))["Close"]
    info = load_info(ticker)
    s0 = float(prices.iloc[-1])
    mu,sigma = calc_gbm_params(prices,lookback_years)
    if use_hybrid:
        mu += 0.5*compute_macro_alpha()*info.get("beta",1) + 0.5*valuation_alpha(info)

    # Input summary
    st.subheader("Input Summary")
    st.markdown(f"- Lookback: {lookback_years} years\n- Paths: {n_paths}\n- Seed: {seed}\n- Target move: {target_move}\n- Hybrid: {'ON' if use_hybrid else 'OFF'}")

    # Forecast table
    rows=[]
    for label,t in HORIZONS.items():
        rows.append({"Horizon":label,**gbm_points(s0,mu,sigma,t)})
    st.dataframe(pd.DataFrame(rows).style.format("${:,.2f}"),use_container_width=True)

    # Monte Carlo mean line only
    paths=simulate_paths(s0,mu,sigma,252,n_paths,seed)
    mean_line=paths.mean(axis=1)
    fig=go.Figure(go.Scatter(x=np.arange(253),y=mean_line,mode="lines",name="Mean Path",line=dict(width=3)))
    fig.update_layout(xaxis_title="Trading Days",yaxis_title="Price")
    st.plotly_chart(fig,use_container_width=True)

    # Target probability
    one_year=HORIZONS["1 Year (252d)"]
    sign=1 if "+" in target_move else -1
    pct=float(target_move.strip("+-").strip("%"))/100
    tgt=s0*(1+sign*pct)
    st.markdown(f"**Hit probability (GBM):** {gbm_prob_reach(s0,mu,sigma,one_year,tgt)*100:.1f}% to reach ${tgt:,.2f} in 1Y")
