import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Forecaster", layout="wide")
st.title("📈 Clean Test Version – No Indentation Issues")

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="5y", auto_adjust=True)
    return df

def gbm_points(s0, mu, sigma, t_years):
    m = math.log(s0) + (mu - 0.5*sigma**2)*t_years
    v = (sigma**2)*t_years
    sd = math.sqrt(v)
    expected = s0 * math.exp(mu * t_years)
    p05, p95 = math.exp(m - 1.96*sd), math.exp(m + 1.96*sd)
    return {"expected": expected, "p05": p05, "p95": p95}

def simulate_paths(s0, mu, sigma, days, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal(0, 1, (days, n_paths))
    paths = np.zeros((days+1, n_paths))
    paths[0] = s0
    drift, vol = (mu - 0.5*sigma**2)*dt, sigma*math.sqrt(dt)
    for t in range(1, days+1):
        paths[t] = paths[t-1] * np.exp(drift + vol * shocks[t-1])
    return paths

ticker = st.text_input("Enter ticker", "AAPL").upper()
run = st.button("Run")

if run:
    st.write(f"✅ App loaded successfully for {ticker}")
    df = load_data(ticker)
    s0 = df["Close"].iloc[-1]
    mu, sigma = 0.1, 0.25
    rows = []
    for label, t in {"1Y": 1}.items():
        rows.append({"Horizon": label, **gbm_points(s0, mu, sigma, t)})
    st.dataframe(pd.DataFrame(rows))
    paths = simulate_paths(s0, mu, sigma, 252, 2000, 42)
    mean_line = paths.mean(axis=1)
    fig = go.Figure(go.Scatter(x=np.arange(253), y=mean_line, mode="lines", line=dict(width=3)))
    st.plotly_chart(fig, use_container_width=True)
