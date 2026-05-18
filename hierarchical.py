import numpy as np
import pandas as pd

def reconcile_forecasts(etf_forecasts, hierarchy_mapping, method="optimal_combination"):
    """
    etf_forecasts: dict {etf: forecasted return}
    hierarchy_mapping: dict {etf: sector}
    Returns reconciled forecasts for each ETF.
    """
    # Group forecasts by sector
    sector_forecasts = {}
    sector_weights = {}
    for etf, fcast in etf_forecasts.items():
        sector = hierarchy_mapping.get(etf, 'Other')
        if sector not in sector_forecasts:
            sector_forecasts[sector] = 0.0
            sector_weights[sector] = 0
        sector_forecasts[sector] += fcast
        sector_weights[sector] += 1
    # Average sector forecast
    for s in sector_forecasts:
        if sector_weights[s] > 0:
            sector_forecasts[s] /= sector_weights[s]
    # Total market forecast (average of sectors)
    total_forecast = np.mean(list(sector_forecasts.values())) if sector_forecasts else 0.0

    if method == "bottom_up":
        # ETF forecasts are unchanged (already bottom‑up)
        reconciled = etf_forecasts.copy()
    elif method == "top_down":
        # Distribute total forecast proportionally to historical weights (equal for simplicity)
        n = len(etf_forecasts)
        reconciled = {etf: total_forecast / n for etf in etf_forecasts}
    else:  # optimal combination (average of bottom‑up and top‑down)
        bottom_up = etf_forecasts
        top_down = {etf: total_forecast / len(etf_forecasts) for etf in etf_forecasts}
        reconciled = {etf: 0.5 * bottom_up[etf] + 0.5 * top_down[etf] for etf in etf_forecasts}
    return reconciled

def evaluate_forecasts(actual_returns, forecasts):
    """Compute MSE for each ETF (optional)."""
    mse = {}
    for etf in actual_returns:
        if etf in forecasts:
            mse[etf] = (actual_returns[etf] - forecasts[etf]) ** 2
    return np.mean(list(mse.values())) if mse else np.inf
