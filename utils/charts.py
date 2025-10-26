# utils/charts.py
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo có cột revenue = qty * unit_price và date là datetime."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # ép kiểu an toàn
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "qty" in out.columns:
        out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0)
    if "unit_price" in out.columns:
        out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce").fillna(0.0)
    if "revenue" not in out.columns and {"qty","unit_price"}.issubset(out.columns):
        out["revenue"] = out["qty"] * out["unit_price"]
    return out.dropna(subset=["date"]) if "date" in out.columns else out

def make_charts_from_orders(orders_df: pd.DataFrame):
    """
    Trả về dict:
      { 'revenue_fig': Figure doanh thu theo ngày,
        'topsku_fig' : Figure top 10 SKU theo doanh thu }
    """
    charts = {"revenue_fig": None, "topsku_fig": None}
    df = _ensure_revenue(orders_df)
    if df is None or df.empty or "revenue" not in df.columns:
        return charts

    # 1) Doanh thu theo ngày (line)
    daily = (df.groupby("date", as_index=False)["revenue"].sum()
               .sort_values("date"))
    fig1, ax1 = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax1.plot(daily["date"], daily["revenue"], marker="o")
    ax1.set_title("Doanh thu theo ngày")
    ax1.set_xlabel("Ngày"); ax1.set_ylabel("Doanh thu (₫)")
    for t in ax1.get_xticklabels():
        t.set_rotation(45); t.set_ha("right")
    fig1.tight_layout()
    charts["revenue_fig"] = fig1

    # 2) Top 10 SKU theo doanh thu (barh)
    top_sku = (df.groupby("sku", as_index=False)["revenue"].sum()
                 .sort_values("revenue", ascending=False)
                 .head(10))
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax2.barh(top_sku["sku"], top_sku["revenue"])
    ax2.invert_yaxis()
    ax2.set_title("Top 10 SKU theo doanh thu")
    ax2.set_xlabel("Doanh thu (₫)"); ax2.set_ylabel("SKU")
    fig2.tight_layout()
    charts["topsku_fig"] = fig2

    return charts
