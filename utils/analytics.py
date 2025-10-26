import pandas as pd

def compute_metrics(orders, inventory):
    orders['qty'] = orders['qty'].astype(float)
    orders['unit_price'] = (orders['unit_price'].astype(str).str.replace(',', '', regex=False).astype(float))
    orders['revenue'] = orders['qty'] * orders['unit_price']
    orders['date'] = pd.to_datetime(orders['date'])

    revenue_by_day = orders.groupby('date')['revenue'].sum().reset_index()
    top_sku = orders.groupby('sku')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(5)

    # avg revenue 7d & 28d
    daily = orders.groupby('date')['revenue'].sum().sort_index()
    avg_7d = daily.tail(7).mean()
    avg_28d = daily.tail(28).mean()

    # tồn kho (days_of_stock)
    sku_sales = orders.groupby('sku')['qty'].sum().reset_index().rename(columns={'qty': 'sold'})
    inv = pd.merge(inventory, sku_sales, on='sku', how='left').fillna(0)
    inv['stock'] = (inv['stock'].astype(str).str.replace(',', '', regex=False).astype(float))
    if 'sold' in inv.columns:inv['sold'] = (inv['sold'].astype(str).str.replace(',', '', regex=False).astype(float))
    else:
        inv['sold'] = 0.0
    inv['days_of_stock'] = inv.apply(
    lambda x: x['stock'] / (x['sold'] / max(len(daily), 1))
    if (isinstance(x['sold'], (int, float)) and x['sold'] > 0)
    else 999,
    axis=1
    )

    return {
        "revenue_by_day": revenue_by_day,
        "top_sku": top_sku,
        "avg_7d": avg_7d,
        "avg_28d": avg_28d,
        "inventory": inv
    }
