# -------------------------------------------------------------
# Stock Forecaster & Valuation Dashboard (Streamlit)
# Final, cleaned, launch-ready version
# -------------------------------------------------------------

import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import norm

# Try to import ARIMA, but keep app working even if statsmodels isn't available
try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

# ---------------------- Constants ----------------------
TRADING_DAYS = 252
HORIZONS = {
    "1 Day": 1/252,
    "5 Days": 5/252,
    "1 Month (~21d)": 21/252,
    "6 Months (~126d)": 126/252,
    "1 Year (252d)": 252/252,
}

# ---------------------- Cache ----------------------
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
    info: Dict = {}
    # Fast info (robust basics)
    try:
        fi = tk.fast_info
        info.update({
            'lastPrice': float(getattr(fi, 'last_price', np.nan)) if getattr(fi, 'last_price', None) is not None else None,
            'marketCap': float(getattr(fi, 'market_cap', np.nan)) if getattr(fi, 'market_cap', None) is not None else None,
            'sharesOutstanding': float(getattr(fi, 'shares_outstanding', np.nan)) if getattr(fi, 'shares_outstanding', None) is not None else None,
        })
    except Exception:
        pass
    # Classic info
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
                info[k] = raw.get(k)
    except Exception:
        pass
    # Try to capture Free Cash Flow from cashflow statement
    try:
        cf = tk.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            idx_norm = [str(x).lower().replace(' ', '') for x in cf.index]
            for i, nm in enumerate(idx_norm):
                if "freecashflow" in nm:
                    series = cf.iloc[i].dropna()
                    if not series.empty:
                        info['freeCashFlow'] = float(series.iloc[0])
                        break
    except Exception:
        pass
    return info

# ---------------------- Math helpers ----------------------
def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
    px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily, sigma_daily = float(rets.mean()), float(rets.std())
    mu_annual, sigma_annual = mu_daily*TRADING_DAYS, sigma_daily*math.sqrt(TRADING_DAYS)
    return mu_annual, sigma_annual

def gbm_distribution(s0: float, mu: float, sigma: float, t_years: float) -> Tuple[float, float]:
    m = math.log(s0) + (mu - 0.5*sigma**2)*t_years
    v = (sigma**2)*t_years
    return m, v

def gbm_points(s0: float, mu: float, sigma: float, t_years: float) -> Dict[str, float]:
    m, v = gbm_distribution(s0, mu, sigma, t_years)
    sd = math.sqrt(v)
    median = math.exp(m)
    expected = s0 * math.exp(mu * t_years)
    p05, p95 = math.exp(m - 1.96*sd), math.exp(m + 1.96*sd)
    return {"expected": expected, "median": median, "p05": p05, "p95": p95}

def gbm_prob_reach(s0: float, mu: float, sigma: float, t_years: float, target: float) -> float:
    if target <= 0:
        return 1.0
    m, v = gbm_distribution(s0, mu, sigma, t_years)
    z = (math.log(target) - m) / math.sqrt(v)
    return float(1 - norm.cdf(z))

def simulate_paths(s0: float, mu: float, sigma: float, days: int, n_paths: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal(0, 1, size=(days, n_paths))
    paths = np.zeros((days+1, n_paths))
    paths[0, :] = s0
    drift, vol = (mu - 0.5*sigma**2)*dt, sigma*math.sqrt(dt)
    for t in range(1, days+1):
        paths[t, :] = paths[t-1, :] * np.exp(drift + vol * shocks[t-1, :])
    return paths

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, window//5)).mean()

# ---------------------- Valuation ----------------------
def heuristic_valuation_flags(info: Dict, prices: pd.Series) -> Tuple[str, Dict[str, str]]:
    reasons = {}
    votes = []
    pe = info.get("trailingPE")
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 12: votes.append(-1); reasons["PE"] = f"Trailing P/E {pe:.1f} (low)"
        elif pe > 30: votes.append(1); reasons["PE"] = f"Trailing P/E {pe:.1f} (high)"
        else: votes.append(0); reasons["PE"] = f"Trailing P/E {pe:.1f} (mid)"
    ps = info.get("priceToSalesTrailing12Months")
    if isinstance(ps, (int, float)) and ps > 0:
        if ps < 2: votes.append(-1); reasons["P/S"] = f"P/S {ps:.1f} (low)"
        elif ps > 10: votes.append(1); reasons["P/S"] = f"P/S {ps:.1f} (very high)"
        else: votes.append(0); reasons["P/S"] = f"P/S {ps:.1f} (mid)"
    pb = info.get("priceToBook")
    if isinstance(pb, (int, float)) and pb > 0:
        if pb < 1: votes.append(-1); reasons["P/B"] = f"P/B {pb:.1f} (below book)"
        elif pb > 6: votes.append(1); reasons["P/B"] = f"P/B {pb:.1f} (high)"
        else: votes.append(0); reasons["P/B"] = f"P/B {pb:.1f} (mid)"
    last = float(prices.iloc[-1])
    ma200 = float(sma(prices, 200).iloc[-1]) if len(prices) >= 200 else None
    if ma200 and ma200 > 0:
        dev = (last/ma200 - 1.0) * 100
        reasons["Trend"] = f"Price is {dev:+.1f}% vs 200-day MA"
        if dev < -10: votes.append(-0.5)
        elif dev > +20: votes.append(0.5)
        else: votes.append(0)
    verdict = "Around Fair Value (heuristic)"
    if votes:
        score = float(np.mean(votes))
        if score <= -0.25: verdict = "Likely Undervalued (heuristic)"
        elif score >= 0.25: verdict = "Likely Overvalued (heuristic)"
    return verdict, reasons

def lightweight_dcf(info: Dict, price: float) -> Optional[Dict[str, float]]:
    fcf, shares, beta = info.get("freeCashFlow"), info.get("sharesOutstanding"), info.get("beta", 1.0)
    if not (isinstance(fcf, (int, float)) and isinstance(shares, (int, float)) and shares > 0):
        return None
    rf, mrp, g = 0.04, 0.055, 0.02
    ke = rf + (beta if isinstance(beta, (int, float)) else 1.0) * mrp
    if ke <= g:
        g = ke - 0.005
    fcf_next = fcf * (1 + g)
    per_share = fcf_next / (ke - g) / shares
    return {"intrinsic": per_share, "premium_vs_price_%": (per_share/price - 1)*100}

# ---------------------- Company Blurb (fixed) ----------------------
def company_position_blurb(info: Dict, prices: pd.Series) -> str:
    """Builds a text summary of the company's current position."""
    pieces = []

    name = info.get("longName") or info.get("shortName") or ""
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    last = float(prices.iloc[-1])

    ma50 = float(sma(prices, 50).iloc[-1]) if len(prices) >= 50 else None
    ma200 = float(sma(prices, 200).iloc[-1]) if len(prices) >= 200 else None
    if ma50:
        pieces.append(f"Price ${last:,.2f} vs 50-day MA ${ma50:,.2f}")
    if ma200:
        pieces.append(f"200-day MA ${ma200:,.2f}")

    rev_g = info.get("revenueGrowth")
    pm = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    dte = info.get("debtToEquity")
    if isinstance(rev_g, (int, float)):
        pieces.append(f"TTM revenue growth {rev_g*100:.1f}%")
    if isinstance(pm, (int, float)):
        pieces.append(f"Profit margin {pm*100:.1f}%")
    if isinstance(roe, (int, float)):
        pieces.append(f"ROE {roe*100:.1f}%")
    if isinstance(dte, (int, float)):
        pieces.append(f"Debt/Equity {dte:.1f}")

    profile_line = ", ".join(p for p in [name, sector, industry] if p)
    summary = info.get("longBusinessSummary")

    blurb = f"**{profile_line}**. " if profile_line else ""
    if pieces:
        blurb += " | ".join(pieces) + ". "
    if summary:
        blurb += f"\n\n**Company Summary:** {summary}"
    return blurb

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Stock Forecaster", layout="wide")
st.title("📈 Stock Forecaster & Valuation Dashboard")
st.caption("This tool is educational. Forecasts are uncertain — not investment advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    lookback_years = st.slider("Return lookback (years)", 1, 10, 3)
    n_paths = st.slider("Monte Carlo paths (1Y)", 500, 10000, 2000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    target_move = st.selectbox("Target move (for hit probability)", ["+10%","+20%","+30%","-10%","-20%","-30%"], index=0)
    run = st.button("Run Analysis", type="primary")

# Always render something to avoid blank screen
st.write("👋 Enter a ticker in the left sidebar and click **Run Analysis**.")

if run and ticker:
    try:
        prices_df = load_history(ticker, years=max(5, lookback_years))
        info = load_info(ticker)
        close = prices_df["Close"]
        s0 = float(close.iloc[-1])
        mu, sigma = calc_gbm_params(close, lookback_years=lookback_years)

        # ---------- Forecast table (GBM) ----------
        rows = []
        for label, t in HORIZONS.items():
            pts = gbm_points(s0, mu, sigma, t)
            rows.append({"Horizon": label, **{k: float(v) for k,v in pts.items()}})
        fc_df = pd.DataFrame(rows)
        st.subheader("Forecasts (GBM, with 95% interval)")
        st.dataframe(fc_df.style.format({"expected":"${:,.2f}","median":"${:,.2f}","p05":"${:,.2f}","p95":"${:,.2f}"}), use_container_width=True)

        # ---------- Optional ARIMA one-year point (if available) ----------
        arima_y = None
        if _HAS_ARIMA:
            try:
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
                if best_model is not None:
                    f = best_model.get_forecast(steps=252)
                    arima_y = float(np.exp(f.predicted_mean.iloc[-1]))
            except Exception:
                arima_y = None

        # ---------- Chart: last 2Y & 1Y forecast points ----------
        st.subheader("Last 2 Years & 1Y Forecast Points")
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
        if arima_y is not None:
            fig.add_trace(go.Scatter(x=[future_date], y=[arima_y], mode='markers', name='ARIMA (1Y)'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Monte Carlo (1Y) ----------
        st.subheader("Monte Carlo (GBM) – 1Y Paths & Distribution")
        paths = simulate_paths(s0, mu, sigma, days=252, n_paths=n_paths, seed=seed)
        sample_idx = np.linspace(0, paths.shape[1]-1, num=min(100, paths.shape[1]), dtype=int)
        df_paths = pd.DataFrame(paths[:, sample_idx])
        df_paths['day'] = np.arange(paths.shape[0])
        df_paths = df_paths.melt(id_vars=['day'], var_name='path', value_name='price')
        fig2 = px.line(df_paths, x='day', y='price', line_group='path', render_mode='webgl')
        st.plotly_chart(fig2, use_container_width=True)

        final_prices = paths[-1, :]
        hist = px.histogram(x=final_prices, nbins=50)
        hist.update_layout(xaxis_title="Price in 1Y", yaxis_title="Frequency")
        st.plotly_chart(hist, use_container_width=True)

        # ---------- Probability of hitting target ----------
        sign = 1 if target_move.startswith('+') else -1
        pct = float(target_move.strip('+-').strip('%'))/100.0
        target_price = s0 * (1 + sign*pct)
        p_gbm = gbm_prob_reach(s0, mu, sigma, one_year, target_price)
        p_mc = float(np.mean(final_prices >= target_price))
        st.markdown(f"**P[S(T) ≥ target ({target_move}, ${target_price:,.2f})]** — GBM: **{p_gbm*100:.1f}%**, Monte Carlo: **{p_mc*100:.1f}%**")

        # ---------- Valuation ----------
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
        else:
            st.caption("DCF not shown (missing FCF or shares outstanding in Yahoo! data).")

        # ---------- Company position ----------
        st.subheader("Company Position")
        st.markdown(company_position_blurb(info, close))

        # ---------- Export CSV ----------
        st.subheader("Export")
        export_df = fc_df.copy()
        export_df["Ticker"] = ticker
        export_df["Spot"] = s0
        export_df["Mu_annual"] = mu
        export_df["Sigma_annual"] = sigma
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecasts CSV", csv, file_name=f"{ticker}_forecasts.csv", mime="text/csv")

    except Exception as e:
        st.error("An error occurred while running the analysis.")
        st.exception(e)
