# -------------------------------------------------------------
# Stock Forecaster — Regime Switching • GARCH-like Vol • ML Drift • Shocks • Diagnostics
# -------------------------------------------------------------
# Educational tool — not investment advice.

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import t as student_t

# Optional ML (auto-disables if scikit-learn unavailable)
ML_AVAILABLE = True
try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:
    ML_AVAILABLE = False
    GradientBoostingRegressor = None

import yfinance as yf

TRADING_DAYS = 252
FORECAST_DAYS = 252     # 1 year paths
ML_HORIZON_DAYS = 21    # next ~1 month for ML target

HORIZONS = {
    "1 Day": 1/252,
    "5 Days": 5/252,
    "1 Month (~21d)": 21/252,
    "6 Months (~126d)": 126/252,
    "1 Year (252d)": 252/252,
}

# ---------------------- Data loaders ----------------------
@st.cache_data(show_spinner=False)
def load_price_history(ticker: str, years: int = 7) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 10)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Could not load price history. Check the ticker.")
    return df.dropna()

@st.cache_data(show_spinner=False)
def load_macro(years: int = 10) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365*years + 10)
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"]  # tenths of pct
    irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"]  # 13-week T-bill (tenths of pct)
    df = pd.DataFrame({"SPY": spy, "VIX": vix, "TNX": tnx, "IRX": irx}).dropna()
    df["TNX_dec"] = df["TNX"]/10.0/100.0
    df["IRX_dec"] = df["IRX"]/10.0/100.0
    df["curve_spread"] = df["TNX_dec"] - df["IRX_dec"]
    df["mom63"] = df["SPY"]/df["SPY"].shift(63) - 1.0
    return df.dropna()

# ---------------------- Stats helpers ----------------------
def calc_gbm_params(prices: pd.Series, lookback_years: int = 3) -> Tuple[float, float]:
    cutoff = prices.index.max() - pd.Timedelta(days=int(lookback_years*365))
    px = prices[prices.index >= cutoff]
    rets = np.log(px/px.shift(1)).dropna()
    mu_daily, sigma_daily = float(rets.mean()), float(rets.std())
    mu_annual = mu_daily * TRADING_DAYS
    sigma_annual = sigma_daily * math.sqrt(TRADING_DAYS)
    return mu_annual, sigma_annual

def gbm_table_rows(s0: float, mu: float, sigma: float) -> pd.DataFrame:
    rows = []
    for label, t in HORIZONS.items():
        expected = s0 * math.exp(mu * t)
        m = math.log(s0) + (mu - 0.5*sigma**2)*t
        sd = math.sqrt((sigma**2)*t)
        p05, p95 = math.exp(m - 1.96*sd), math.exp(m + 1.96*sd)
        rows.append({"Horizon": label, "expected": expected, "p05": p05, "p95": p95})
    return pd.DataFrame(rows)

# ---------------------- Regimes ----------------------
def regime_probabilities(vix: float, mom63: float, curve_spread: float) -> Dict[str, float]:
    # Soft rule-based weights → probabilities
    w_bull = 1.0; w_neutral = 1.0; w_bear = 1.0
    if vix < 15: w_bull += 1.5
    elif vix < 25: w_neutral += 0.5
    else: w_bear += 1.5
    if mom63 > 0.10: w_bull += 1.0
    elif mom63 < -0.05: w_bear += 1.0
    if curve_spread < 0: w_bear += 0.5
    total = w_bull + w_neutral + w_bear
    return {"Bull": w_bull/total, "Neutral": w_neutral/total, "Bear": w_bear/total}

def regime_params(regime: str) -> Tuple[float, float]:
    # (mu_annual, sigma_annual) baselines by regime
    if regime == "Bull":   return 0.12, 0.18
    if regime == "Bear":   return -0.08, 0.35
    return 0.06, 0.24  # Neutral

# ---------------------- ML short-horizon drift ----------------------
def build_ml_features(px: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"Close": px}).dropna()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["ret21"] = df["Close"].pct_change(21)
    df["vol21"] = df["ret1"].rolling(21).std()
    df = df.join(macro[["VIX","mom63","curve_spread"]], how="left")
    df["target21"] = np.log(df["Close"].shift(-ML_HORIZON_DAYS)/df["Close"])
    return df.dropna()

def fit_ml_model(feat: pd.DataFrame) -> Optional[GradientBoostingRegressor]:
    if not ML_AVAILABLE or feat.shape[0] < 200:
        return None
    X = feat[["ret1","ret5","ret21","vol21","VIX","mom63","curve_spread"]].values
    y = feat["target21"].values
    split = max(100, int(0.85*len(feat)))
    model = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X[:split], y[:split])
    return model

def predict_ml_mu(feat_latest: pd.Series, model: Optional[GradientBoostingRegressor]) -> Optional[float]:
    if model is None:
        return None
    x = np.array([[
        feat_latest["ret1"], feat_latest["ret5"], feat_latest["ret21"], feat_latest["vol21"],
        feat_latest["VIX"], feat_latest["mom63"], feat_latest["curve_spread"]
    ]])
    pred_21 = float(model.predict(x)[0])
    return pred_21 * (TRADING_DAYS / ML_HORIZON_DAYS)

# ---------------------- Simulation engine ----------------------
def simulate_regime_garch_paths(
    s0: float,
    mu_base: float,
    sigma_hist: float,
    macro_latest: Dict[str, float],
    ml_mu: Optional[float],
    n_paths: int = 3000,
    days: int = FORECAST_DAYS,
    seed: int = 42,
    p_persist: float = 0.92,
    omega: float = 2e-5, alpha: float = 0.08, beta: float = 0.90,
    shock_prob: float = 0.05, shock_df: int = 5, shock_scale: float = 0.20
) -> Tuple[np.ndarray, Dict]:
    """Regime-switching with GARCH-like daily variance and occasional downward shocks."""
    rng = np.random.default_rng(seed)

    probs = regime_probabilities(macro_latest["VIX"], macro_latest["mom63"], macro_latest["curve_spread"])
    regimes = ["Bull","Neutral","Bear"]
    reg_params_map = {r: regime_params(r) for r in regimes}

    paths = np.zeros((days+1, n_paths))
    paths[0] = s0

    current_reg = rng.choice(regimes, size=n_paths, p=[probs["Bull"], probs["Neutral"], probs["Bear"]])
    # Initialize daily variance from historical sigma
    var_t = np.ones(n_paths) * (sigma_hist/ math.sqrt(TRADING_DAYS))**2

    blend_ml = 0.3 if (ml_mu is not None) else 0.0
    dt = 1.0/TRADING_DAYS

    for t in range(1, days+1):
        # Regime persistence / switching
        switch_mask = rng.random(n_paths) > p_persist
        if switch_mask.any():
            current_reg[switch_mask] = rng.choice(regimes, size=int(switch_mask.sum()), p=[probs["Bull"], probs["Neutral"], probs["Bear"]])

        mu_reg_annual = np.array([reg_params_map[r][0] for r in current_reg])
        sig_reg_annual = np.array([reg_params_map[r][1] for r in current_reg])

        mu_adj_annual = mu_base + (1.0 - blend_ml)*mu_reg_annual + blend_ml*(ml_mu if ml_mu is not None else 0.0)
        drift_daily = (mu_adj_annual - 0.5*(sig_reg_annual**2)) * dt

        # GARCH-like variance update
        eps_prev = rng.standard_normal(n_paths) * np.sqrt(var_t)
        var_t = omega + alpha*(eps_prev**2) + beta*var_t
        var_t = np.maximum(var_t, 1e-8)
        vol_daily = np.sqrt(var_t)

        # Downward fat-tail shocks
        shock_mult = np.ones(n_paths)
        shock_mask = rng.random(n_paths) < shock_prob
        if shock_mask.any():
            draws = student_t.rvs(df=shock_df, size=int(shock_mask.sum()), random_state=rng)
            drops = -np.abs(draws) * shock_scale
            shock_mult[shock_mask] = np.exp(drops)

        z = rng.standard_normal(n_paths)
        incr = drift_daily + vol_daily * z
        paths[t] = paths[t-1] * np.exp(incr) * shock_mult

    details = {"probs": probs, "reg_params": reg_params_map}
    return paths, details

# ---------------------- Diagnostics (quick walk-forward) ----------------------
def diagnostics_walkforward(
    prices: pd.Series,
    macro: pd.DataFrame,
    lookback_years: int,
    checkpoints: int = 12,
    paths_per_check: int = 400,
    seed: int = 123
) -> pd.DataFrame:
    # Use roughly monthly checkpoints over ~1y
    px = prices.dropna()
    if px.shape[0] < ML_HORIZON_DAYS + 30:
        return pd.DataFrame()

    # Restrict to last ~year
    end_idx = px.index[-1]
    start_idx = end_idx - pd.Timedelta(days=365)
    px = px[px.index >= start_idx]
    if px.shape[0] < ML_HORIZON_DAYS + 30:
        return pd.DataFrame()

    step = max(ML_HORIZON_DAYS, int(px.shape[0]/checkpoints))
    rows = []
    rng_seed = seed

    for i in range(0, px.shape[0] - ML_HORIZON_DAYS, step):
        t0 = px.index[i]
        t1 = px.index[min(i + ML_HORIZON_DAYS, px.shape[0]-1)]
        s0 = float(prices.loc[:t0].iloc[-1])
        s1 = float(prices.loc[:t1].iloc[-1])

        hist_up_to = prices.loc[:t0]
        mu_base, sigma_hist = calc_gbm_params(hist_up_to, lookback_years=lookback_years)

        macro_up_to = macro.loc[:t0].iloc[-1]
        macro_latest = {"VIX": float(macro_up_to["VIX"]), "mom63": float(macro_up_to["mom63"]), "curve_spread": float(macro_up_to["curve_spread"])}

        feat = build_ml_features(hist_up_to, macro.loc[:t0])
        model = fit_ml_model(feat)
        ml_mu = predict_ml_mu(feat.iloc[-1], model) if (model is not None and not feat.empty) else None

        paths, _ = simulate_regime_garch_paths(
            s0, mu_base, sigma_hist, macro_latest, ml_mu,
            n_paths=paths_per_check, days=ML_HORIZON_DAYS, seed=rng_seed,
            shock_prob=0.05, shock_df=5, shock_scale=0.20
        )
        rng_seed += 1

        final = paths[-1]
        p05, p95 = float(np.percentile(final, 5)), float(np.percentile(final, 95))
        mean = float(np.mean(final))
        covered = (min(p05, p95) <= s1 <= max(p05, p95))
        rel_err = (mean - s1) / s1
        rows.append({"t0": t0, "t1": t1, "start": s0, "actual": s1, "mean": mean, "p05": p05, "p95": p95, "covered": covered, "rel_error": rel_err})

    return pd.DataFrame(rows)

# ---------------------- UI ----------------------
st.set_page_config(page_title="Stock Forecaster (Realistic)", layout="wide")
st.title("📉📈 Stock Forecaster — Regimes • GARCH-like Vol • ML Drift • Shocks")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    lookback_years = st.slider("Return lookback (years)", 1, 10, 3)

    st.subheader("Realism Layers")
    use_regimes = st.checkbox("Regime switching", True)
    use_garch = st.checkbox("GARCH-like volatility", True)
    use_ml = st.checkbox("ML drift adjuster (next ~21d)", True if ML_AVAILABLE else False)
    if not ML_AVAILABLE:
        st.caption("⚠️ scikit-learn not found — ML drift disabled. Install scikit-learn to enable.")

    st.subheader("Shocks")
    use_shocks = st.checkbox("Macro shock events (fat tails)", True)
    shock_prob = st.slider("Shock probability (daily)", 0.00, 0.20, 0.05, 0.01)
    shock_scale = st.slider("Shock scale (severity)", 0.00, 0.50, 0.20, 0.01)

    st.subheader("Monte Carlo")
    n_paths = st.slider("Paths", 500, 10000, 3000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

tabs = st.tabs(["Forecast", "Diagnostics"])

with tabs[0]:
    st.write("Enter a ticker and click **Run Forecast**. The chart displays the **mean** of all simulated paths (no spaghetti).")
    if st.button("Run Forecast", type="primary") and ticker:
        try:
            prices_df = load_price_history(ticker, years=max(7, lookback_years))
            close = prices_df["Close"]
            s0 = float(close.iloc[-1])
            mu_base, sigma_hist = calc_gbm_params(close, lookback_years=lookback_years)

            macro = load_macro()
            mlast = macro.iloc[-1]
            macro_latest = {
                "VIX": float(mlast["VIX"]),
                "mom63": float(mlast["mom63"]),
                "curve_spread": float(mlast["curve_spread"]),
            }

            # Optional ML drift
            ml_mu = None
            if use_ml and ML_AVAILABLE:
                feat = build_ml_features(close, macro)
                model = fit_ml_model(feat)
                if model is not None and not feat.empty:
                    ml_mu = predict_ml_mu(feat.iloc[-1], model)

            # Switches to disable pieces by parameterization
            p_persist = 0.92 if use_regimes else 1.0
            omega = 2e-5 if use_garch else 0.0
            alpha = 0.08 if use_garch else 0.0
            beta = 0.90 if use_garch else 0.0
            sp = shock_prob if use_shocks else 0.0
            ss = shock_scale if use_shocks else 0.0

            paths, details = simulate_regime_garch_paths(
                s0=s0,
                mu_base=mu_base,
                sigma_hist=sigma_hist,
                macro_latest=macro_latest,
                ml_mu=ml_mu if (use_ml and ML_AVAILABLE) else None,
                n_paths=n_paths,
                days=FORECAST_DAYS,
                seed=int(seed),
                p_persist=p_persist,
                omega=omega, alpha=alpha, beta=beta,
                shock_prob=sp, shock_df=5, shock_scale=ss
            )

            # Forecast table (quick GBM approximation with base mu & hist sigma)
            st.subheader("Forecasts (approximate, 95% interval) — GBM baseline")
            fc_df = gbm_table_rows(s0, mu_base, sigma_hist)
            num_cols = [c for c in fc_df.columns if c != "Horizon"]
            st.dataframe(fc_df.style.format({c: "${:,.2f}" for c in num_cols}), use_container_width=True)

            # Monte Carlo mean line
            st.subheader("Monte Carlo — Mean Path (1Y)")
            mean_line = paths.mean(axis=1)
            fig = go.Figure(go.Scatter(x=np.arange(paths.shape[0]), y=mean_line, mode="lines", name="Mean path"))
            fig.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Regime details"):
                st.write({"probabilities": details["probs"], "regime_params": details["reg_params"]})

        except Exception as e:
            st.error("An error occurred while running the forecast.")
            st.exception(e)

with tabs[1]:
    st.write("A quick walk-forward check over the last ~year using monthly checkpoints.")
    if st.button("Run Diagnostics (fast)") and ticker:
        try:
            prices_df = load_price_history(ticker, years=max(7, lookback_years))
            macro = load_macro()
            close = prices_df["Close"]
            diag = diagnostics_walkforward(close, macro, lookback_years=lookback_years, checkpoints=12, paths_per_check=400)
            if diag.empty:
                st.info("Not enough data to run diagnostics.")
            else:
                coverage = 100.0 * float(diag["covered"].mean())
                rmse = float(np.sqrt(np.mean((diag["mean"] - diag["actual"])**2)))
                st.markdown(f"**Coverage (within 5–95% band): {coverage:.1f}%**")
                st.markdown(f"**RMSE of mean forecast: ${rmse:,.2f}**")
                st.dataframe(diag, use_container_width=True)
        except Exception as e:
            st.error("Diagnostics failed.")
            st.exception(e)
