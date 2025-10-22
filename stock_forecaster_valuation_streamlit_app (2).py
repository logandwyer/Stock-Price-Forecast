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
#   • Walk‑forward backtest, peer comps, CSV export
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
            'lastPrice': float(getattr(fi, 'last_price', np.nan)) if getattr(fi, 'last_price', None) is not None else None,
            'marketCap': float(getattr(fi, 'market_cap', np.nan)) if getattr(fi, 'market_cap', None) is not None else None,
            'sharesOutstanding': float(getattr(fi, 'shares_outstanding', np.nan)) if getattr(fi, 'shares_outstanding', None) is not None else None,
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
    # Try to pull free cash flow for DCF
    try:
        cf = tk.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            # Look for a free cash flow row name variant
            names = [str(x).lower().replace(' ', '') for x in cf.index]
            for i,name in enumerate(names):
                if name in ['freecashflow','freecashflow','freecashflow']:
                    series = cf.iloc[i].dropna()
                    if not series.empty:
                        info['freeCashFlow'] = float(series.iloc[0])
                        break
    except Exception:
        pass
    return info

# ---------------------- Math helpers ----------------------

def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    # Restrict to lookback window when available
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
        reasons['Trend'] = f"Price is {dev:+.1f}% vs 200-day MA"
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
        pieces.append(f"Price ${last:,.2f} vs 50-day MA ${ma50:,.2f}")
    if ma200:
        pieces.append(f"200-day MA ${ma200:,.2f}")
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
        blurb += f"

**Company Summary:** {summary}"
    return blurb

# ---------------------- Backtest ----------------------

def backtest(close: pd.Series, lookback_years: int, method: str = 'GBM', horizon_days: int = 1, test_days: int = 60) -> Dict[str, float]:
    """Walk-forward multi-step ahead backtest over last test_days.
    Returns RMSE, MAPE, Directional Accuracy, N.
    """
    assert method in ['GBM','ARIMA','Ensemble']
    if len(close) < 300:
        return {"RMSE": np.nan, "MAPE%": np.nan, "Directional%": np.nan, "N": 0}
    actuals = []
    preds = []
    s0s = []
    idx = close.index
    start_idx = len(close) - test_days - horizon_days
    start_idx = max(start_idx, 260)
    for i in range(start_idx, len(close) - horizon_days):
        train = close.iloc[:i+1]
        cutoff = train.index.max() - pd.Timedelta(days=int(lookback_years*365))
        train = train[train.index >= cutoff]
        if len(train) < 120:
            continue
        s0_i = float(train.iloc[-1])
        # GBM
        mu_i, sigma_i = calc_gbm_params(train, lookback_years=min(lookback_years, 3))
        pred_gbm = gbm_points(s0_i, mu_i, sigma_i, horizon_days/252)['expected']
        # ARIMA
        pred_arima = np.nan
        try:
            log_px = np.log(train)
            best_aic = np.inf
            best_model = None
            for p in range(0,3):
                for q in range(0,3):
                    try:
                        res = ARIMA(log_px, order=(p,1,q)).fit(method_kwargs={"warn_convergence":False})
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_model = res
                    except Exception:
                        continue
            if best_model is not None:
                f = best_model.get_forecast(steps=horizon_days)
                pred_arima = float(np.exp(f.predicted_mean.iloc[-1]))
        except Exception:
            pass
        if method == 'GBM':
            pred = pred_gbm
        elif method == 'ARIMA':
            pred = pred_arima
        else:
            pred = pred_gbm if np.isnan(pred_arima) else 0.5*pred_gbm + 0.5*pred_arima
        if np.isnan(pred):
            continue
        actual = float(close.iloc[i + horizon_days])
        actuals.append(actual)
        preds.append(pred)
        s0s.append(s0_i)
    if not preds:
        return {"RMSE": np.nan, "MAPE%": np.nan, "Directional%": np.nan, "N": 0}
    preds = np.array(preds)
    actuals = np.array(actuals)
    s0s = np.array(s0s)
    rmse = float(np.sqrt(np.mean((preds - actuals)**2)))
    mape = float(np.mean(np.abs((actuals - preds)/actuals))) * 100
    dir_hits = np.sign(preds - s0s) == np.sign(actuals - s0s)
    da = float(np.mean(dir_hits)) * 100
    return {"RMSE": rmse, "MAPE%": mape, "Directional%": da, "N": int(len(preds))}

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Stock Forecaster & Valuation", layout="wide")

st.title("📈 Stock Forecaster & Valuation (Upgraded)")
st.caption("Enter a ticker. The app pulls public data and estimates 1D, 5D, 1M, 6M, 1Y prices via GBM and ARIMA, shows Monte Carlo paths, and provides valuation snapshots (heuristic + optional DCF). Educational only, not investment advice.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    lookback_years = st.slider("Return lookback (years)", 1, 10, 3)
    n_paths = st.slider("Monte Carlo paths (1Y)", 500, 10000, 3000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("**DCF assumptions**")
    st.session_state['rf_rate'] = st.number_input("Risk‑free rate (rf)", min_value=0.0, max_value=0.15, value=0.04, step=0.005, format="%.3f")
    st.session_state['mkt_premium'] = st.number_input("Market premium (MRP)", min_value=0.0, max_value=0.20, value=0.055, step=0.005, format="%.3f")
    st.session_state['terminal_g'] = st.number_input("Terminal growth g", min_value=0.0, max_value=0.06, value=0.02, step=0.002, format="%.3f")
    target_move = st.selectbox("Target move (for hit probability)", ["+10%","+20%","+30%","-10%","-20%","-30%"], index=0)
    st.markdown("**Backtest**")
    bt_enable = st.checkbox("Run backtest (1‑day ahead)", value=True)
    bt_days = st.slider("Test window (days)", 20, 120, 60)
    st.markdown("**Peer comps**")
    comps_str = st.text_input("Peer tickers (comma‑separated)", value="")
    run = st.button("Run Analysis", type="primary")

if run and ticker:
    try:
        prices_df = load_history(ticker, years=max(5, lookback_years))
        info = load_info(ticker)
        close = prices_df['Close']
        s0 = float(close.iloc[-1])
        mu, sigma = calc_gbm_params(close, lookback_years=lookback_years)

        # --------- ARIMA model (grid search p,q in [0..2]) ---------
        st.subheader("Forecasts")
        horizon_days = {"1 Day":1, "5 Days":5, "1 Month (~21d)":21, "6 Months (~126d)":126, "1 Year (252d)":252}
        log_px = np.log(close)
        best_aic = np.inf
        best_model = None
        for p in range(0,3):
            for q in range(0,3):
                try:
                    res = ARIMA(log_px, order=(p,1,q)).fit(method_kwargs={"warn_convergence":False})
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                except Exception:
                    continue
        arima_forecasts = {}
        if best_model is not None:
            for label, days in horizon_days.items():
                f = best_model.get_forecast(steps=days)
                mean_log = f.predicted_mean.iloc[-1]
                arima_forecasts[label] = float(np.exp(mean_log))
        else:
            arima_forecasts = {k: np.nan for k in horizon_days.keys()}

        # --------- GBM analytics & table (with ensemble) ---------
        rows = []
        for label, t in HORIZONS.items():
            pts = gbm_points(s0, mu, sigma, t)
            arima_px = arima_forecasts.get(label, np.nan)
            ensemble = 0.5*pts['expected'] + 0.5*arima_px if not np.isnan(arima_px) else pts['expected']
            rows.append({
                "Horizon": label,
                "GBM Expected": pts['expected'],
                "ARIMA Point": arima_px,
                "Ensemble": ensemble,
                "Median": pts['median'],
                "Low (2.5%)": pts['p05'],
                "High (97.5%)": pts['p95']
            })
        fc_df = pd.DataFrame(rows)
        st.dataframe(
            fc_df.style.format({c: "${:,.2f}" for c in ["GBM Expected","ARIMA Point","Ensemble","Median","Low (2.5%)","High (97.5%)"]}),
            use_container_width=True
        )

        # --------- Chart: last 2Y and 1Y forecast points ---------
        end = close.index[-1]
        start = end - pd.Timedelta(days=365*2)
        vis = close[close.index >= start]
        one_year = HORIZONS["1 Year (252d)"]
        band = gbm_points(s0, mu, sigma, one_year)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vis.index, y=vis.values, mode='lines', name='Close'))
        future_date = vis.index[-1] + pd.Timedelta(days=365)
        fig.add_trace(go.Scatter(x=[future_date], y=[band['expected']], mode='markers', name='GBM Expected (1Y)'))
        fig.add_trace(go.Scatter(x=[future_date], y=[band['p05']], mode='markers', name='Low 2.5% (1Y)'))
        fig.add_trace(go.Scatter(x=[future_date], y=[band['p95']], mode='markers', name='High 97.5% (1Y)'))
        if not np.isnan(arima_forecasts.get("1 Year (252d)", np.nan)):
            fig.add_trace(go.Scatter(x=[future_date], y=[arima_forecasts["1 Year (252d)"]], mode='markers', name='ARIMA (1Y)'))
        fig.update_layout(title=f"{ticker} – Last 2Y & 1Y Forecast Points", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # --------- Monte Carlo paths (1Y) ---------
        st.subheader("Monte Carlo (GBM) – 1Y Paths & Distribution")
        paths = simulate_paths(s0, mu, sigma, days=252, n_paths=n_paths, seed=seed)
        sample_idx = np.linspace(0, paths.shape[1]-1, num=min(100, paths.shape[1]), dtype=int)
        df_paths = pd.DataFrame(paths[:, sample_idx])
        df_paths['day'] = np.arange(paths.shape[0])
        df_paths = df_paths.melt(id_vars=['day'], var_name='path', value_name='price')
        fig2 = px.line(df_paths, x='day', y='price', line_group='path', render_mode='webgl')
        fig2.update_layout(title="Sample of Monte Carlo Paths (1Y)", xaxis_title="Trading Days Ahead", yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)

        final_prices = paths[-1, :]
        hist = px.histogram(x=final_prices, nbins=50)
        hist.update_layout(title="Distribution of Simulated 1Y Prices", xaxis_title="Price in 1Y", yaxis_title="Frequency")
        st.plotly_chart(hist, use_container_width=True)

        # --------- Probability of hit target ---------
        sign = 1 if target_move.startswith('+') else -1
        pct = float(target_move.strip('+-%'))/100.0
        target_price = s0 * (1 + sign*pct)
        p_gbm = gbm_prob_reach(s0, mu, sigma, one_year, target_price)
        p_mc = float(np.mean(final_prices >= target_price))
        st.markdown(f"**Probability S(T) ≥ {target_move} target (price ${target_price:,.2f})** — GBM closed‑form: **{p_gbm*100:.1f}%**, Monte Carlo: **{p_mc*100:.1f}%**")

        # --------- Valuation ---------
        st.subheader("Valuation Snapshot")
        verdict, reasons = heuristic_valuation_flags(info, close)
        st.markdown(f"**Heuristic Verdict:** {verdict}")
        if reasons:
            with st.expander("Why this verdict?"):
                for k, v in reasons.items():
                    st.write(f"• {k}: {v}")

        dcf = lightweight_dcf(info, s0)
        if dcf:
            st.markdown(f"**Lightweight DCF per‑share:** ${dcf['intrinsic']:,.2f}  (premium vs price: {dcf['premium_vs_price_%']:.1f}%)")
            with st.expander("DCF details"):
                st.write({k: (f"{v:.4f}" if isinstance(v, float) else v) for k,v in dcf.items()})
        else:
            st.caption("DCF not shown (missing FCF or shares outstanding in Yahoo! data).")

        # --------- Company position ---------
        st.subheader("Company Position")
        st.markdown(company_position_blurb(info, close))

        # --------- Fundamentals table ---------
        with st.expander("Raw fundamentals (selected)"):
            keys = [
                'sector','industry','country','trailingPE','forwardPE',
                'priceToBook','priceToSalesTrailing12Months','dividendYield','beta',
                'profitMargins','grossMargins','operatingMargins','returnOnEquity',
                'revenueGrowth','earningsGrowth','debtToEquity','currentRatio','marketCap','sharesOutstanding'
            ]
            rows = []
            for k in keys:
                val = info.get(k, None)
                if isinstance(val, (int, float)):
                    if 'Margins' in k or 'margin' in k or k in ['revenueGrowth','earningsGrowth','dividendYield']:
                        val = f"{val*100:.2f}%"
                    elif k in ['marketCap']:
                        val = f"${val:,.0f}"
                    elif k in ['sharesOutstanding']:
                        val = f"{val:,.0f}"
                    else:
                        val = f"{val:.4g}"
                rows.append((k, val))
            df_info = pd.DataFrame(rows, columns=['Field','Value'])
            st.dataframe(df_info, use_container_width=True)

        # --------- Backtest ---------
        if bt_enable:
            st.subheader("Backtest (walk‑forward, 1‑day ahead)")
            col1, col2, col3 = st.columns(3)
            with col1:
                bt_gbm = backtest(close, lookback_years, method='GBM', horizon_days=1, test_days=bt_days)
                st.metric("GBM RMSE", f"${bt_gbm['RMSE']:.2f}")
                st.metric("GBM MAPE", f"{bt_gbm['MAPE%']:.1f}%")
                st.metric("GBM Directional", f"{bt_gbm['Directional%']:.1f}%")
            with col2:
                bt_arima = backtest(close, lookback_years, method='ARIMA', horizon_days=1, test_days=bt_days)
                st.metric("ARIMA RMSE", f"${bt_arima['RMSE']:.2f}")
                st.metric("ARIMA MAPE", f"{bt_arima['MAPE%']:.1f}%")
                st.metric("ARIMA Directional", f"{bt_arima['Directional%']:.1f}%")
            with col3:
                bt_ens = backtest(close, lookback_years, method='Ensemble', horizon_days=1, test_days=bt_days)
                st.metric("Ensemble RMSE", f"${bt_ens['RMSE']:.2f}")
                st.metric("Ensemble MAPE", f"{bt_ens['MAPE%']:.1f}%")
                st.metric("Ensemble Directional", f"{bt_ens['Directional%']:.1f}%")
            st.caption("ARIMA refits each step with (p,q) in 0..2; results are indicative only.")

        # --------- Peer Comps ---------
        st.subheader("Peer Comps (manual list)")
        if comps_str.strip():
            peers = [x.strip().upper() for x in comps_str.split(',') if x.strip()]
            comp_rows = []
            for peer in peers:
                try:
                    inf = load_info(peer)
                    comp_rows.append({
                        'Ticker': peer,
                        'Sector': inf.get('sector'),
                        'Industry': inf.get('industry'),
                        'Market Cap': inf.get('marketCap'),
                        'P/E (TTM)': inf.get('trailingPE'),
                        'P/E (Fwd)': inf.get('forwardPE'),
                        'P/B': inf.get('priceToBook'),
                        'P/S (TTM)': inf.get('priceToSalesTrailing12Months'),
                        'Profit Margin %': (inf.get('profitMargins')*100 if isinstance(inf.get('profitMargins'), (int,float)) else None),
                        'ROE %': (inf.get('returnOnEquity')*100 if isinstance(inf.get('returnOnEquity'), (int,float)) else None),
                        'Rev Growth %': (inf.get('revenueGrowth')*100 if isinstance(inf.get('revenueGrowth'), (int,float)) else None)
                    })
                except Exception:
                    continue
            if comp_rows:
                df_comps = pd.DataFrame(comp_rows)
                if 'Market Cap' in df_comps:
                    df_comps['Market Cap'] = df_comps['Market Cap'].map(lambda x: f"${x:,.0f}" if isinstance(x,(int,float)) else x)
                for c in ['Profit Margin %','ROE %','Rev Growth %']:
                    if c in df_comps:
                        df_comps[c] = df_comps[c].map(lambda x: f"{x:.1f}%" if isinstance(x,(int,float)) else x)
                st.dataframe(df_comps, use_container_width=True)
            else:
                st.caption("No peer data could be retrieved.")
        else:
            st.caption("Provide comma‑separated tickers in the sidebar to see comps (e.g., MSFT, GOOGL, AMZN).")

        # --------- Export CSV ---------
        st.subheader("Export")
        export_df = fc_df.copy()
        export_df['Ticker'] = ticker
        export_df['Spot'] = s0
        export_df['Mu_annual'] = mu
        export_df['Sigma_annual'] = sigma
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecasts CSV", csv, file_name=f"{ticker}_forecasts.csv", mime="text/csv")

        st.info("This tool is educational. Forecasts are statistical and uncertain. Always do your own research.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.write("Enter a ticker in the sidebar and click **Run Analysis** to begin.")

# ----------------------
# requirements.txt (create alongside this file):
# streamlit
# yfinance
# pandas
# numpy
# plotly
# statsmodels
# scipy
