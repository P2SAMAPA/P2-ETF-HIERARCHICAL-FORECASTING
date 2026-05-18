import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, universe_tickers):
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    returns = returns.dropna(how='all')
    return returns

def load_hierarchy(universe_tickers):
    """
    Load or create a hierarchy mapping: ETF -> sector.
    For simplicity, we'll create a dummy mapping based on ticker prefixes.
    In production, a CSV file would be used.
    """
    # Try to load from CSV
    try:
        hier = pd.read_csv(config.HIERARCHY_FILE, index_col=0)
        mapping = hier.to_dict().get('sector', {})
        return mapping
    except FileNotFoundError:
        # Create simple mapping based on ticker prefixes
        mapping = {}
        for t in universe_tickers:
            if t in ['TLT','VCIT','LQD','HYG','VNQ']:
                sector = 'FI'
            elif t in ['GLD','SLV']:
                sector = 'Commodity'
            elif t in ['SPY','QQQ','XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','IWF','IWD','IWO','IWM','XSD','XBI','GDX','XME']:
                sector = 'Equity'
            else:
                sector = 'Other'
            mapping[t] = sector
        return mapping
