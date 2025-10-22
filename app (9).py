# -------------------------------------------------------------
# Stock Forecaster & Valuation Dashboard — Hybrid (Real‑World Enhanced)
# Features:
#  - Input Summary (lookback, paths, seed, target move, hybrid toggle)
#  - Forecast table for 1D, 5D, 1M, 6M, 1Y (GBM with adjusted drift)
#  - Monte Carlo (1Y) mean-only line
#  - Probability of hitting target move
#  - Heuristic valuation + lightweight DCF
#  - Company position blurb (trend + fundamentals)
#  - CSV export
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
            "lastPrice": float(getattr(fi, "last_price", np.nan)) if getattr(fi, "last_price", None) is not None else None,
            "marketCap": float(getattr(fi, "market_cap", np.nan)) if getattr(fi, "market_cap", None) is not None else None,
            "sharesOutstanding": float(getattr(fi, "shares_outstanding", np.nan)) if getattr(fi, "shares_outstanding", None) is not None else None,
        })
    except Exception:
        pass
    # Classic info (keys may be partial)
    try:
        raw = tk.info
        if isinstance(raw, dict):
            for k in [
                "shortName","longName","industry","sector","country","website",
                "trailingPE","forwardPE","priceToBook","priceToSalesTrailing12Months",
                "dividendYield","beta","profitMargins","grossMargins","operatingMargins",
                "debtToEquity","returnOnEquity","revenueGrowth","earningsGrowth","currentRatio",
                "longBusinessSummary"
            ]:
                info[k] = raw.get(k)
    except Exception:
        pass
    # Cash flow (try to find Free Cash Flow)
    try:
        cf = tk.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            idx_norm = [str(x).lower().replace(" ", "") for x in cf.index]
            for i, nm in enumerate(idx_norm):
                if "freecashflow" in nm:
                    series = cf.iloc[i].dropna()
                    if not series.empty:
                        info["freeCashFlow"] = float(series.iloc[0])
                        break
    except Exception:
        pass
    return info

# ---------------------- Math helpers ----------------------
def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
    px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily = float(rets.mean())
    sigma_daily = float(rets.std())
    mu_annual = mu_daily * TRADING_DAYS
    sigma_annual = sigma_daily * math.sqrt(TRADING_DAYS)
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
    p05 = math.exp(m - 1.9599639845*sd)
    p95 = math.exp(m + 1.9599639845*sd)
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
    drift = (mu - 0.5*sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    for t in range(1, days+1):
        paths[t, :] = paths[t-1, :] * np.exp(drift + vol * shocks[t-1, :])
    return paths

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, window//5)).mean()

# ---------------------- Valuation ----------------------
def heuristic_valuation_flags(info: Dict, prices: pd.Series) -> Tuple[str, Dict[str, str]]:
    reasons: Dict[str, str] = {}
    votes = []
    pe = info.get("trailingPE", None)
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 12:
            votes.append(-1); reasons["PE"] = f"Trailing P/E {pe:.1f} (low)"
        elif pe > 30:
            votes.append(1); reasons["PE"] = f"Trailing P/E {pe:.1f} (high)"
        else:
            votes.append(0); reasons["PE"] = f"Trailing P/E {pe:.1f} (mid)"
    ps = info.get("priceToSalesTrailing12Months", None)
    if isinstance(ps, (int, float)) and ps > 0:
        if ps < 2:
            votes.append(-1); reasons["P/S"] = f"P/S {ps:.1f} (low)"
        elif ps > 10:
            votes.append(1); reasons["P/S"] = f"P/S {ps:.1f} (very high)"
        else:
            votes.append(0); reasons["P/S"] = f"P/S {ps:.1f} (mid)"
    pb = info.get("priceToBook", None)
    if isinstance(pb, (int, float)) and pb > 0:
        if pb < 1:
            votes.append(-1); reasons["P/B"] = f"P/B {pb:.1f} (below book)"
        elif pb > 6:
            votes.append(1); reasons["P/B"] = f"P/B {pb:.1f} (high)"
        else:
            votes.append(0); reasons["P/B"] = f"P/B {pb:.1f} (mid)"
    last = float(prices.iloc[-1])
    ma200 = float(sma(prices, 200).iloc[-1]) if len(prices) >= 200 else None
    if ma200 and ma200 > 0:
        dev = (last/ma200 - 1.0) * 100
        reasons["Trend"] = f"Price is {dev:+.1f}% vs 200-day MA"
        if dev < -10: votes.append(-0.5)
        elif dev > +20: votes.append(0.5)
        else: votes.append(0)
    if not votes:
        return ("Inconclusive (insufficient data)", reasons)
    score = float(np.mean(votes))
    if score <= -0.25:
        verdict = "Likely Undervalued (heuristic)"
    elif score >= 0.25:
        verdict = "Likely Overvalued (heuristic)"
    else:
        verdict = "Around Fair Value (heuristic)"
    return verdict, reasons

def lightweight_dcf(info: Dict, price: float) -> Optional[Dict[str, float]]:
    fcf = info.get("freeCashFlow", None)
    shares = info.get("sharesOutstanding", None)
    beta = info.get("beta", 1.0) or 1.0
    if not (isinstance(fcf, (int, float)) and isinstance(shares, (int, float)) and shares > 0):
        return None
    rf = 0.04
    mrp = 0.055
    g = 0.02
    ke = rf + (beta if isinstance(beta, (int, float)) else 1.0) * mrp
    if ke <= g:
        g = ke - 0.005
    fcf_next = fcf * (1 + g)
    equity_value = fcf_next / (ke - g)
    per_share = equity_value / shares
    return {
        "rf": rf, "mrp": mrp, "beta": beta, "ke": ke, "g": g,
        "fcf_ttm": fcf, "shares": shares, "intrinsic": per_share,
        "premium_vs_price_%": (per_share/price - 1) * 100
    }

# ---------------------- Company Blurb ----------------------
def company_position_blurb(info: Dict, prices: pd.Series) -> str:
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
    profile_line = ", ".join([p for p in [name, sector, industry] if p])
    summary = info.get("longBusinessSummary")
    blurb = f"**{profile_line}**. " if profile_line else ""
    if pieces:
        blurb += " | ".join(pieces) + ". "
    if summary:
        blurb += f"\n\n**Company Summary:** {summary}"
    return blurb

# ---------------------- Macro alpha via quick OLS on SPY/VIX/TNX ----------------------
@st.cache_data(show_spinner=False)
def compute_macro_alpha() -> Tuple[Optional[float], Dict[str, float]]:
    """Estimate next-21d market drift using VIX level, 10Y yield, and 63d momentum.
    Returns (annualized_alpha, details dict). If unavailable, returns (None, {}).
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=365*10)
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
        vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
        tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"]  # tenths of a percent
        df = pd.DataFrame({"SPY": spy, "VIX": vix, "TNX": tnx}).dropna().copy()
        df["TNX_dec"] = df["TNX"] / 10.0 / 100.0  # e.g., 45 -> 4.5% -> 0.045
        df["mom63"] = df["SPY"] / df["SPY"].shift(63) - 1.0
        df["fwd21"] = df["SPY"].shift(-21) / df["SPY"] - 1.0
        df = df.dropna()
        X = np.column_stack([np.ones(len(df)), df["VIX"].values, df["TNX_dec"].values, df["mom63"].values])
        y = df["fwd21"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        V = float(df["VIX"].iloc[-1]); T = float(df["TNX_dec"].iloc[-1]); M = float(df["mom63"].iloc[-1])
        pred_21d = float(beta[0] + beta[1]*V + beta[2]*T + beta[3]*M)
        alpha_annual = pred_21d * (TRADING_DAYS/21.0)
        details = {
            "beta0": float(beta[0]), "beta_VIX": float(beta[1]), "beta_TNX": float(beta[2]), "beta_mom63": float(beta[3]),
            "VIX": V, "TNX_dec": T, "mom63": M, "pred_21d": pred_21d, "alpha_annual": alpha_annual
        }
        return alpha_annual, details
    except Exception:
        return None, {}

def valuation_alpha(info: Dict) -> float:
    """Small valuation tilt in annualized return based on simple thresholds."""
    adj = 0.0
    pe = info.get("trailingPE", None)
    ps = info.get("priceToSalesTrailing12Months", None)
    pb = info.get("priceToBook", None)
    if isinstance(pe, (int, float)) and pe > 0:
        if pe < 12: adj += 0.02
        elif pe > 30: adj -= 0.02
    if isinstance(ps, (int, float)) and ps > 0:
        if ps < 2: adj += 0.01
        elif ps > 10: adj -= 0.01
    if isinstance(pb, (int, float)) and pb > 0:
        if pb < 1: adj += 0.01
        elif pb > 6: adj -= 0.01
    return adj

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Stock Forecaster (Hybrid)", layout="wide")
st.title("📈 Stock Forecaster & Valuation — Hybrid (Real‑World Enhanced)")
st.caption("Educational only — not investment advice. Forecasts are uncertain.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    lookback_years = st.slider("Return lookback (years)", 1, 10, 3)
    use_hybrid = st.checkbox("Use hybrid (real‑world) adjustments", value=True)
    n_paths = st.slider("Monte Carlo paths (1Y)", 500, 10000, 2000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    target_move = st.selectbox("Target move (for hit probability)", ["+10%","+20%","+30%","-10%","-20%","-30%"], index=0)
    run = st.button("Run Analysis", type="primary")

st.write("👋 Enter a ticker on the left, choose whether to use **hybrid (real‑world) adjustments**, and click **Run Analysis**.")

if run and ticker:
    try:
        prices_df = load_history(ticker, years=max(5, lookback_years))
        info = load_info(ticker)
        close = prices_df["Close"]
        s0 = float(close.iloc[-1])
        mu_base, sigma = calc_gbm_params(close, lookback_years=lookback_years)

        # ---------- Hybrid adjustments ----------
        macro_ann, macro_details = compute_macro_alpha() if use_hybrid else (None, {})
        beta_stock = info.get("beta", 1.0) or 1.0
        alpha_macro_stock = (macro_ann or 0.0) * (beta_stock if isinstance(beta_stock, (int, float)) else 1.0)
        alpha_val = valuation_alpha(info) if use_hybrid else 0.0
        mu_adj = mu_base + 0.5*alpha_macro_stock + 0.5*alpha_val  # simple blend

        # ---------- Input Summary ----------
        st.subheader("Input Summary")
        st.markdown(
            f"- **Return lookback:** {lookback_years} years  \n"
            f"- **Monte Carlo paths:** {n_paths}  \n"
            f"- **Random seed:** {seed}  \n"
            f"- **Target move:** {target_move}  \n"
            f"- **Hybrid adjustments:** {'ON' if use_hybrid else 'OFF'}"
        )

        # ---------- Mu decomposition ----------
        with st.expander("Hybrid μ decomposition"):
            st.write({
                "mu_base_annual": mu_base,
                "macro_alpha_annual (market)": macro_ann,
                "beta_stock": beta_stock,
                "macro_alpha_scaled (stock)": alpha_macro_stock,
                "valuation_alpha_annual": alpha_val,
                "mu_adjusted_annual": mu_adj
            })
            if macro_details:
                st.caption("Macro model details (OLS on SPY next‑21d returns):")
                st.write(macro_details)

        # ---------- Forecast table (GBM with adjusted mu) ----------
        rows = []
        for label, t in HORIZONS.items():
            pts = gbm_points(s0, mu_adj if use_hybrid else mu_base, sigma, t)
            rows.append({"Horizon": label, **{k: float(v) for k, v in pts.items()}})
        fc_df = pd.DataFrame(rows)
        st.subheader("Forecasts (GBM with adjusted drift, 95% interval)")
        num_cols = [c for c in fc_df.columns if c != "Horizon"]
        st.dataframe(fc_df.style.format({c: "${:,.2f}" for c in num_cols}), use_container_width=True)

        # ---------- Monte Carlo (1Y) mean-only line ----------
        st.subheader("Monte Carlo (GBM) — 1Y Mean Path")
        paths = simulate_paths(s0, mu_adj if use_hybrid else mu_base, sigma, days=252, n_paths=n_paths, seed=seed)
        mean_line = paths.mean(axis=1)
        fig = go.Figure(go.Scatter(x=np.arange(paths.shape[0]), y=mean_line, mode="lines", name="Mean of simulations", line=dict(width=3)))
        fig.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Probability of hitting target ----------
        one_year = HORIZONS["1 Year (252d)"]
        sign = 1 if target_move.startswith("+") else -1
        pct = float(target_move.strip("+-").strip("%"))/100.0
        target_price = s0 * (1 + sign*pct)
        p_gbm = gbm_prob_reach(s0, (mu_adj if use_hybrid else mu_base), sigma, one_year, target_price)
        # Monte Carlo probability (use distribution of final simulated prices)
        final_prices = paths[-1, :]
        p_mc = float(np.mean(final_prices >= target_price)) if sign > 0 else float(np.mean(final_prices <= target_price))
        st.markdown(f"**P[S(T) {'≥' if sign>0 else '≤'} target ({target_move}, ${target_price:,.2f})]** — GBM: **{p_gbm*100:.1f}%**, Monte Carlo: **{p_mc*100:.1f}%**")

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
            with st.expander("DCF details"):
                st.write({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in dcf.items()})
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
        export_df["Mu_base_annual"] = mu_base
        export_df["Mu_adjusted_annual"] = mu_adj if use_hybrid else mu_base
        export_df["Sigma_annual"] = sigma
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecasts CSV", csv, file_name=f"{ticker}_forecasts.csv", mime="text/csv")

    except Exception as e:
        st.error("An error occurred while running the analysis.")
        st.exception(e)
