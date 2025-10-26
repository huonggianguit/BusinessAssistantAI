# utils/planner.py
import pandas as pd
import numpy as np

def build_promo_plan(inv_df: pd.DataFrame, dos_threshold=60, default_discount=12):
    """
    Nhận inv_df (đã có: sku, name, stock, qty_7d, qty_14d, avg_daily_14, days_of_stock)
    -> trả về promo_plan (DataFrame) + thống kê nhỏ
    """
    if inv_df is None or inv_df.empty:
        return pd.DataFrame(), {"num_sku": 0}

    df = inv_df.copy()
    # Làm tròn hiển thị (giữ tính toán bên trong là float đã ok)
    df["days_of_stock"] = pd.to_numeric(df["days_of_stock"], errors="coerce")
    df["days_of_stock"] = np.ceil(df["days_of_stock"]).astype("Int64")

    plan = df[df["days_of_stock"] > dos_threshold].copy()
    if plan.empty:
        return plan, {"num_sku": 0}

    # Quy tắc giảm giá theo tồn/ngày
    def promo_rule(dos):
        if dos >= 120: return max(default_discount, 25)
        if dos >= 90:  return max(default_discount, 20)
        if dos >= 60:  return max(default_discount, 15)
        return default_discount

    plan["discount_%"] = plan["days_of_stock"].apply(promo_rule)
    plan["mechanic"]   = np.where(plan["discount_%"] >= 20, "Flash 48h + Bundle", "Code giảm giá 7 ngày")
    cols = ["sku","name","stock","qty_7d","qty_14d","avg_daily_14","days_of_stock","discount_%","mechanic"]
    plan = plan[[c for c in cols if c in plan.columns]].sort_values(["days_of_stock","stock"], ascending=False)

    return plan, {"num_sku": len(plan)}
