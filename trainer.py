import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import config
import data_manager
from hierarchical import reconcile_forecasts

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_features(returns_df, macro_df, window):
    """Simple features: lagged returns and macro levels."""
    ret_win = returns_df.iloc[-window:]
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret_win.index, columns=config.MACRO_COLUMNS)
    common = ret_win.index.intersection(macro_win.index)
    ret_win = ret_win.loc[common]
    macro_win = macro_win.loc[common]
    # Features: recent return (1d), 5d avg, 21d std, macro levels
    X_list = []
    for col in ret_win.columns:
        ret = ret_win[col].values.reshape(-1,1)
        ret_5d = ret_win[col].rolling(5).mean().values.reshape(-1,1)
        ret_21d_std = ret_win[col].rolling(21).std().values.reshape(-1,1)
        macro_vals = macro_win.values
        X = np.hstack([ret, ret_5d, ret_21d_std, macro_vals])
        X_list.append(X)
    # Stack horizontally (n_samples, n_etfs * n_features) – not good. We'll use separate models per ETF.
    return None  # We'll handle per ETF in training loop

def train_etf_model(returns_df, macro_df, etf, window):
    """
    Train a simple random forest to predict next day's return for a single ETF.
    Features: past returns of the ETF (lag 1,5,21) + macro levels.
    """
    ret = returns_df[etf].iloc[-window:]
    macro = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret.index, columns=config.MACRO_COLUMNS)
    common = ret.index.intersection(macro.index)
    ret = ret.loc[common]
    macro = macro.loc[common]
    # Create features
    df = pd.DataFrame(index=ret.index)
    df['ret1'] = ret
    df['ret5'] = ret.rolling(5).mean()
    df['ret21_std'] = ret.rolling(21).std()
    for mc in macro.columns:
        df[mc] = macro[mc]
    df = df.dropna()
    if len(df) < 30:
        return None
    X = df.drop(columns=['ret1']).values  # features except current return
    y = df['ret1'].shift(-1).dropna().values
    X = X[:-1]
    if len(X) < 10:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42)
    model.fit(X_scaled, y)
    # Predict for the most recent feature vector
    last_X = X[-1:].reshape(1, -1)
    last_scaled = scaler.transform(last_X)
    pred = model.predict(last_scaled)[0]
    return pred

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Hierarchical Forecasting) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        # Load hierarchy mapping
        hierarchy = data_manager.load_hierarchy(tickers)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 20:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            # Get base forecasts for each ETF (using simple random forest)
            base_forecasts = {}
            for etf in tickers:
                pred = train_etf_model(returns, macro, etf, win)
                if pred is not None:
                    base_forecasts[etf] = pred
            if not base_forecasts:
                continue
            # Reconcile forecasts using hierarchical method
            reconciled = reconcile_forecasts(base_forecasts, hierarchy, method=config.METHOD)
            window_results[win] = reconciled
            for etf, score in reconciled.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "forecast": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by reconciled forecast: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/hierarchical_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Hierarchical Forecasting Engine complete ===")

if __name__ == "__main__":
    main()
