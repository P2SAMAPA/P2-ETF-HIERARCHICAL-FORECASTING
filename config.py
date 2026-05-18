import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-hierarchical-forecast-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Hierarchy levels (based on sector classification)
# For simplicity, we'll predefine a mapping: ETF -> sector
# We'll auto‑generate from existing sector ETFs? Not needed.
# We'll assume the user provides a mapping CSV or we derive from names.
# Here we'll use a simple hard‑coded mapping for the three universes.
# For demonstration, we'll create a mapping file locally. The engine will load if present.
HIERARCHY_FILE = "etf_hierarchy.csv"

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# Forecasting method: "bottom_up" or "top_down" or "optimal_combination"
METHOD = "optimal_combination"

TOP_N = 3
