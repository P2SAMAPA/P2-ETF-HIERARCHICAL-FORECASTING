# Hierarchical Forecasting Engine

Reconciles ETF return forecasts across multiple levels (ETF → sector → total market) using bottom‑up, top‑down, or optimal combination methods. Base forecasts come from a random forest using macro features and lagged returns. Multi‑window evaluation selects the best window per ETF.

- **Hierarchy:** sector mapping derived from ticker prefixes (can be customised)
- **Reconciliation methods:** bottom‑up, top‑down, optimal combination
- **Base model:** Random Forest (macro + lagged returns)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by reconciled forecast

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
