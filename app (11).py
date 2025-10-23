# -------------------------------------------------------------
# Stock Forecaster — Regime-Switching, GARCH-like Vol, ML Drift, Shocks, Diagnostics
# -------------------------------------------------------------
# Features
# - Hybrid drift: GBM base + regime switching + ML drift adjuster
# - Regimes (Bull/Neutral/Bear) inferred from VIX, SPY momentum, yield-curve spread
# - GARCH-like conditional volatility during simulation
# - Macro shocks (fat-tailed, downward only) with adjustable probability
# - Mean-only Monte Carlo line (per prior request)
# - Heuristic valuation, lightweight DCF, company blurb
# - Diagnostics tab: simple walk-forward (monthly) coverage and error stats
#
# Educational tool — not investment advice.

import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, t as student_t
ML_AVAILABLE = True
try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    ML_AVAILABLE = False
    GradientBoostingRegressor = None

# ---------------------- Constants ----------------------
TRADING_DAYS = 252
HORIZONS = {
    "1 Day": 1/252,
    "5 Days": 5/252,
    "1 Month (~21d)": 21/252,
    "6 Months (~126d)": 126/252,
    "1 Year (252d)": 252/252,
}
FORECAST_DAYS = 252  # for MC paths
ML_HORIZON_DAYS = 21  # next-month target for ML

# ---------------------- Caching ----------------------
@st.cache_data(show_spinner=False)
def load_history(ticker: str, years: int = 7) -> pd.DataFrame:
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
    try:
        fi = tk.fast_info
        info.update({
            "lastPrice": float(getattr(fi, "last_price", np.nan)) if getattr(fi, "last_price", None) is not None else None,
            "marketCap": float(getattr(fi, "market_cap", np.nan)) if getattr(fi, "market_cap", None) is not None else None,
            "sharesOutstanding": float(getattr(fi, "shares_outstanding", np.nan)) if getattr(fi, "shares_outstanding", None) is not None else None,
        })
    except Exception:
        pass
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

@st.cache_data(show_spinner=False)
def load_macro(years: int = 10) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 10)
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"]  # tenths of pct
    irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"]  # 13-week T-bill (tenths of pct)
    df = pd.DataFrame({"SPY": spy, "VIX": vix, "TNX": tnx, "IRX": irx}).dropna()
    # Features
    df["TNX_dec"] = df["TNX"] / 10.0 / 100.0
    df["IRX_dec"] = df["IRX"] / 10.0 / 100.0
    df["curve_spread"] = df["TNX_dec"] - df["IRX_dec"]
    df["mom63"] = df["SPY"] / df["SPY"].shift(63) - 1.0
    df = df.dropna()
    return df

# ---------------------- Helpers ----------------------
def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
    px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily = float(rets.mean())
    sigma_daily = float(rets.std())
    mu_annual = mu_daily * TRADING_DAYS
    sigma_annual = sigma_daily * math.sqrt(TRADING_DAYS)
    return mu_annual, sigma_annual

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, window//5)).mean()

# ---------------------- Regimes ----------------------
def regime_probabilities(vix: float, mom63: float, curve_spread: float) -> Dict[str, float]:
    """Return probs for Bull/Neutral/Bear (soft rules turned into weights)."""
    # Start with neutral weights
    w_bull = 1.0
    w_neutral = 1.0
    w_bear = 1.0
    # VIX
    if vix < 15: w_bull += 1.5
    elif vix < 25: w_neutral += 0.5
    else: w_bear += 1.5
    # Momentum
    if mom63 > 0.10: w_bull += 1.0
    elif mom63 < -0.05: w_bear += 1.0
    # Yield curve
    if curve_spread < 0: w_bear += 0.5
    total = w_bull + w_neutral + w_bear
    return {"Bull": w_bull/total, "Neutral": w_neutral/total, "Bear": w_bear/total}

def regime_params(regime: str) -> Tuple[float, float]:
    """Return (mu_regime_annual, sigma_regime_annual) baseline for regime."""
    if regime == "Bull":
        return 0.12, 0.18
    if regime == "Bear":
        return -0.08, 0.35
    return 0.06, 0.24  # Neutral

# ---------------------- ML Drift Adjuster ----------------------
def build_ml_features(px: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"Close": px}).dropna()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["ret21"] = df["Close"].pct_change(21)
    df["vol21"] = df["ret1"].rolling(21).std()
    df = df.join(macro[["VIX","mom63","curve_spread"]], how="left")
    df = df.dropna()
    # Target: forward 21d log return
    df["target21"] = np.log(df["Close"].shift(-ML_HORIZON_DAYS) / df["Close"])
    df = df.dropna()
    return df

def fit_ml_model(feat: pd.DataFrame) -> Optional[GradientBoostingRegressor]:
    if feat.shape[0] < 200:
        return None
    X = feat[["ret1","ret5","ret21","vol21","VIX","mom63","curve_spread"]].values
    y = feat["target21"].values
    # Train on all but last 100 samples to reduce overfit a bit
    split = max(100, int(0.85 * len(feat)))
    model = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X[:split], y[:split])
    return model

def predict_ml_mu(feat_latest: pd.Series, model: Optional[GradientBoostingRegressor]) -> Optional[float]:
    if model is None:
        return None
    x = np.array([[
        feat_latest["ret1"], feat_latest["ret5"], feat_latest["ret21"],
        feat_latest["vol21"], feat_latest["VIX"], feat_latest["mom63"], feat_latest["curve_spread"]
    ]])
    pred_21 = float(model.predict(x)[0])
    mu_ml_annual = pred_21 * (TRADING_DAYS / ML_HORIZON_DAYS)
    return mu_ml_annual

# ---------------------- Valuation & DCF ----------------------
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

# ---------------------- Simulation Engine ----------------------
def simulate_regime_garch_paths(
    s0: float,
    mu_base: float,
    sigma_hist: float,
    macro_latest: Dict[str, float],
    ml_mu: Optional[float],
    n_paths: int = 2000,
    days: int = FORECAST_DAYS,
    seed: int = 42,
    p_persist: float = 0.90,
    omega: float = 0.00002, alpha: float = 0.08, beta: float = 0.90,
    shock_prob: float = 0.05, shock_df: int = 5, shock_scale: float = 0.20
) -> Tuple[np.ndarray, Dict]:
    """Regime-switching with GARCH-like vol and occasional negative shocks."""
    rng = np.random.default_rng(seed)

    # Regime probabilities from macro
    probs = regime_probabilities(macro_latest["VIX"], macro_latest["mom63"], macro_latest["curve_spread"])
    regimes = ["Bull","Neutral","Bear"]
    # Precompute regime params
    reg_params = {r: regime_params(r) for r in regimes}

    # Initialize arrays
    paths = np.zeros((days+1, n_paths))
    paths[0,:] = s0
    # Start regime per path
    current_reg = rng.choice(regimes, size=n_paths, p=[probs["Bull"], probs["Neutral"], probs["Bear"]])
    # Initialize conditional variance with hist sigma
    var_t = np.ones(n_paths) * (sigma_hist/ math.sqrt(TRADING_DAYS))**2  # daily variance

    # ML blend
    blend_ml = 0.3 if ml_mu is not None else 0.0

    for t in range(1, days+1):
        # Possibly switch regime per path with persistence
        switch_mask = rng.random(n_paths) > p_persist
        if switch_mask.any():
            new_states = rng.choice(regimes, size=int(switch_mask.sum()), p=[probs["Bull"], probs["Neutral"], probs["Bear"]])
            current_reg[switch_mask] = new_states

        # Regime params for each path
        mu_reg_annual = np.array([reg_params[r][0] for r in current_reg])
        sig_reg_annual = np.array([reg_params[r][1] for r in current_reg])

        # Adjust drift: base + (1-blend)*regime + blend*ml
        mu_adj_annual = mu_base + (1.0 - blend_ml) * mu_reg_annual + blend_ml * (ml_mu if ml_mu is not None else 0.0)

        # Convert to daily drift
        dt = 1.0 / TRADING_DAYS
        drift_daily = (mu_adj_annual - 0.5 * (sig_reg_annual**2)) * dt

        # GARCH-like vol update (daily variance)
        eps_prev = rng.standard_normal(n_paths) * np.sqrt(var_t)
        var_t = omega + alpha * (eps_prev**2) + beta * var_t
        # Bound variance away from zero
        var_t = np.maximum(var_t, 1e-8)
        vol_daily = np.sqrt(var_t)

        # Shocks (downward only)
        shock_mask = rng.random(n_paths) < shock_prob
        shock_mult = np.ones(n_paths)
        if shock_mask.any():
            draws = student_t.rvs(df=shock_df, size=int(shock_mask.sum()), random_state=rng)
            drops = -np.abs(draws) * shock_scale  # negative only
            shock_mult[shock_mask] = np.exp(drops)

        # Step
        z = rng.standard_normal(n_paths)
        incr = drift_daily + vol_daily * z
        paths[t,:] = paths[t-1,:] * np.exp(incr) * shock_mult

    details = {"probs": probs, "reg_params": reg_params}
    return paths, details

# ---------------------- Diagnostics ----------------------
def diagnostics_walkforward(
    prices: pd.Series,
    macro: pd.DataFrame,
    lookback_years: int,
    checkpoints: int = 12,
    paths_per_check: int = 500,
    seed: int = 123
) -> pd.DataFrame:
    """Simple monthly walk-forward over the last ~year."""
    # Pick checkpoint dates (roughly monthly)
    end_idx = prices.index[-1]
    start_idx = end_idx - pd.Timedelta(days=365)
    px = prices[prices.index >= start_idx].copy()
    if px.shape[0] < ML_HORIZON_DAYS + 30:
        return pd.DataFrame()

    step = max(ML_HORIZON_DAYS, int(px.shape[0] / checkpoints))
    rows = []
    rng_seed = seed
    for i in range(0, px.shape[0] - ML_HORIZON_DAYS, step):
        t0 = px.index[i]
        t1 = px.index[min(i + ML_HORIZON_DAYS, px.shape[0]-1)]
        s0 = float(prices.loc[:t0].iloc[-1])
        s1 = float(prices.loc[:t1].iloc[-1])

        # Fit params using data up to t0
        hist_up_to = prices.loc[:t0]
        mu_base, sigma_hist = calc_gbm_params(hist_up_to, lookback_years=lookback_years)

        # Macro snapshot at t0 (use nearest)
        macro_up_to = macro.loc[:t0].iloc[-1]
        macro_latest = {"VIX": float(macro_up_to["VIX"]), "mom63": float(macro_up_to["mom63"]), "curve_spread": float(macro_up_to["curve_spread"])}

        # ML at t0
        feat = build_ml_features(hist_up_to, macro.loc[:t0])
        model = fit_ml_model(feat)
        ml_mu = None
        if not feat.empty:
            ml_mu = predict_ml_mu(feat.iloc[-1], model)

        # Simulate fewer paths for speed
        paths, _ = simulate_regime_garch_paths(
            s0, mu_base, sigma_hist, macro_latest, ml_mu,
            n_paths=paths_per_check, days=ML_HORIZON_DAYS, seed=rng_seed,
            shock_prob=0.05, shock_df=5, shock_scale=0.20
        )
        rng_seed += 1
        final = paths[-1,:]
        # 5-95 band and expected (mean)
        p05 = float(np.percentile(final, 5))
        p95 = float(np.percentile(final, 95))
        mean = float(np.mean(final))
        covered = (min(p05, p95) <= s1 <= max(p05, p95))
        err = (mean - s1) / s1
        rows.append({"t0": t0, "t1": t1, "start": s0, "actual": s1, "mean": mean, "p05": p05, "p95": p95, "covered": covered, "rel_error": err})
    return pd.DataFrame(rows)

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Stock Forecaster — Regime + GARCH + ML + Shocks", layout="wide")
st.title("📉📈 Stock Forecaster — Regime-Switching • GARCH-like Vol • ML Drift • Shocks")
st.caption("Educational only — not investment advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    lookback_years = st.slider("Return lookback (years)", 1, 10, 3)

    st.subheader("Realism Layers")
    use_regimes = st.checkbox("Regime switching (Bull/Neutral/Bear)", True)
    use_garch = st.checkbox("GARCH-like volatility updates", True)
    \1
    if not ML_AVAILABLE:
        st.warning('ML drift disabled: scikit-learn not available. Add scikit-learn to requirements.txt or use the provided one.'); use_ml = False", True)
    use_shocks = st.checkbox("Macro shock events (fat tails)", True)

    st.subheader("Monte Carlo")
    n_paths = st.slider("Paths", 500, 10000, 3000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.subheader("Shock Settings")
    shock_prob = st.slider("Shock probability (daily)", 0.00, 0.20, 0.05, 0.01)
    shock_scale = st.slider("Shock scale (approx severity)", 0.00, 0.50, 0.20, 0.01)

    go_btn = st.button("Run Forecast", type="primary")

tabs = st.tabs(["Forecast", "Diagnostics"])

with tabs[0]:
    st.write("Enter a ticker and click **Run Forecast**. Mean-only Monte Carlo line is displayed by design.")

    if go_btn and ticker:
        try:
            prices_df = load_history(ticker, years=max(7, lookback_years))
            info = load_info(ticker)
            macro = load_macro()

            close = prices_df["Close"]
            s0 = float(close.iloc[-1])
            mu_base, sigma_hist = calc_gbm_params(close, lookback_years=lookback_years)

            # Macro snapshot (latest)
            mrow = macro.iloc[-1]
            macro_latest = {"VIX": float(mrow["VIX"]), "mom63": float(mrow["mom63"]), "curve_spread": float(mrow["curve_spread"])}

            # ML drift
            ml_mu = None
            if use_ml:
                feat = build_ml_features(close, macro)
                model = fit_ml_model(feat)
                if not feat.empty:
                    ml_mu = predict_ml_mu(feat.iloc[-1], model)

            # Regime switching + GARCH + Shocks
            paths, details = simulate_regime_garch_paths(
                s0=s0,
                mu_base=mu_base,
                sigma_hist=sigma_hist,
                macro_latest=macro_latest,
                ml_mu=ml_mu if use_ml else None,
                n_paths=n_paths,
                days=FORECAST_DAYS,
                seed=int(seed),
                p_persist=0.92 if use_regimes else 1.0,  # 1.0 → no switching
                omega=0.00002 if use_garch else 0.0,
                alpha=0.08 if use_garch else 0.0,
                beta=0.90 if use_garch else 0.0,
                shock_prob=shock_prob if use_shocks else 0.0,
                shock_df=5,
                shock_scale=shock_scale if use_shocks else 0.0
            )

            # Forecast summary table (from GBM points using approximate mu, sigma)
            st.subheader("Forecasts (approximate, 95% interval)")
            rows = []
            for label, t in HORIZONS.items():
                # Use base mu & hist sigma for the quick table (simulation is authoritative)
                exp = s0 * math.exp(mu_base * t)
                m = math.log(s0) + (mu_base - 0.5*sigma_hist**2)*t
                sd = math.sqrt((sigma_hist**2)*t)
                p05, p95 = math.exp(m - 1.96*sd), math.exp(m + 1.96*sd)
                rows.append({"Horizon": label, "expected": exp, "p05": p05, "p95": p95})
            fc_df = pd.DataFrame(rows)
            num_cols = [c for c in fc_df.columns if c != "Horizon"]
            st.dataframe(fc_df.style.format({c: "${:,.2f}" for c in num_cols}), use_container_width=True)

            # Monte Carlo mean line only
            st.subheader("Monte Carlo — 1Y Mean Path (Regime/GARCH/Shocks)")
            mean_line = paths.mean(axis=1)
            fig = go.Figure(go.Scatter(x=np.arange(paths.shape[0]), y=mean_line, mode="lines", name="Mean path", line=dict(width=3)))
            fig.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Probability of hitting +/− target (use +10% for example? Keep your prior control simple)
            st.subheader("Target Analysis")
            target_move = st.selectbox("Target move for 1Y", ["+10%","+20%","+30%","-10%","-20%","-30%"], index=0)
            sign = 1 if target_move.startswith("+") else -1
            pct = float(target_move.strip("+-").strip("%"))/100.0
            target_price = s0 * (1 + sign*pct)
            final_prices = paths[-1,:]
            p_mc = float(np.mean(final_prices >= target_price)) if sign > 0 else float(np.mean(final_prices <= target_price))
            st.markdown(f"**P[Price {'≥' if sign>0 else '≤'} target in 1Y] (Monte Carlo): {p_mc*100:.1f}%** (target ${target_price:,.2f})")

            # Valuation snapshot & DCF
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

            # Company position
            st.subheader("Company Position")
            st.markdown(company_position_blurb(info, close))

            # Export
            st.subheader("Export")
            export_df = fc_df.copy()
            export_df["Ticker"] = ticker
            export_df["Spot"] = s0
            export_df["Mu_base_annual"] = mu_base
            export_df["Sigma_annual"] = sigma_hist
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Forecasts CSV", csv, file_name=f"{ticker}_forecasts.csv", mime="text/csv")

            # Explain regimes
            with st.expander("Regime details"):
                st.write({"probs": details["probs"], "params": details["reg_params"]})

        except Exception as e:
            st.error("An error occurred while running the forecast.")
            st.exception(e)

with tabs[1]:
    st.write("A light walk-forward check over the last ~year using monthly checkpoints.")
    if st.button("Run Diagnostics (fast)") and ticker:
        try:
            prices_df = load_history(ticker, years=max(7, lookback_years))
            macro = load_macro()
            close = prices_df["Close"]
            diag = diagnostics_walkforward(close, macro, lookback_years=lookback_years, checkpoints=12, paths_per_check=400)
            if diag.empty:
                st.info("Not enough data to run diagnostics.")
            else:
                coverage = 100.0 * float(diag["covered"].mean())
                rmse = float(np.sqrt(np.mean(((diag["mean"] - diag["actual"])**2))))
                st.markdown(f"**Coverage (within 5–95% band): {coverage:.1f}%**")
                st.markdown(f"**RMSE of mean forecast: ${rmse:,.2f}**")
                st.dataframe(diag, use_container_width=True)
        except Exception as e:
            st.error("Diagnostics failed.")
            st.exception(e)
