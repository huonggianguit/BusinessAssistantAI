# main.py — Business Assistant AI (Agent MVP)
import os, re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick 
import matplotlib.dates as mdates
from dotenv import load_dotenv

# === utils gốc của bạn ===
from utils.sheets import read_sheet
from utils.analytics import compute_metrics
#from utils.charts import make_charts
from utils.charts import make_charts_from_orders
from utils.llm import gen_insights
from utils.planner import build_promo_plan
# === agent tools ===
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO

# ----------------- ENV / PAGE -----------------
load_dotenv()
st.set_page_config(page_title="Business Assistant AI", layout="wide")
st.title("🤖 Business Assistant AI")
st.sidebar.header("Cấu hình")

SHEET_ID_DEFAULT = os.getenv("SHEET_ID", "")
CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# ----------------- HELPERS -----------------
def _gc(scopes):
    if not CREDS_PATH or not os.path.exists(CREDS_PATH):
        raise RuntimeError("Thiếu GOOGLE_APPLICATION_CREDENTIALS hoặc đường dẫn không tồn tại.")
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
    return gspread.authorize(creds)

def _df_safe_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    # Ép kiểu để gspread/JSON không lỗi NAType/Int64
    out = df.copy().astype(object)           # bỏ Int64/BooleanDtype...
    out = out.where(pd.notna(out), None)     # pd.NA/NaN/NaT -> None
    out.columns = [str(c) for c in out.columns]  # header là string
    return out

def write_promo_plan_to_sheet(sheet_id: str, promo_df: pd.DataFrame) -> str:
    # luôn chuẩn hoá DF trước
    safe = _df_safe_for_sheets(promo_df)

    gc = _gc(["https://www.googleapis.com/auth/spreadsheets"])
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet("promo_plan")
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(
            title="promo_plan",
            rows=max(100, safe.shape[0] + 10),
            cols=max(10,  safe.shape[1] + 2),
        )

    # ghi dữ liệu
    ws.update([safe.columns.tolist()] + safe.values.tolist())
    return "Đã ghi tab 'promo_plan' ✅"

def read_agent_state(sheet_id: str) -> dict:
    """Memory: đọc các key/value trong tab 'agent_state' (nếu có)."""
    try:
        gc = _gc(["https://www.googleapis.com/auth/spreadsheets.readonly"])
        ws = gc.open_by_key(sheet_id).worksheet("agent_state")
        rows = ws.get_all_records()
        return {str(r.get("key","")).strip(): str(r.get("value","")).strip() for r in rows if r}
    except Exception:
        return {}

def write_agent_state(sheet_id: str, kv: dict) -> str:
    """Memory: ghi key/value vào tab 'agent_state'."""
    gc = _gc(["https://www.googleapis.com/auth/spreadsheets"])
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("agent_state"); ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title="agent_state", rows=max(50, len(kv)+5), cols=2)
    data = [["key","value"]] + [[k, str(v)] for k,v in kv.items()]
    ws.update(data)
    return "Đã lưu cấu hình agent ✅"

def clean_bullets(text: str) -> list[str]:
    text = str(text or "").strip()
    text = re.sub(r"[*•_#]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(1️⃣|2️⃣|3️⃣)", r"\n\1 ", text)
    parts = re.split(r"(?:\n|^)(?=[123]️⃣)", text)
    parts = [p.strip() for p in parts if p and "BÁO CÁO" not in p.upper()]
    return parts

def to_number(s):
    return pd.to_numeric(
        pd.Series(s, dtype="object").astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)  # bỏ ₫, ký tự
        .str.replace(",", "", regex=False)          # bỏ ,
        .str.replace(".", "", regex=False),         # bỏ . nếu là phân cách nghìn
        errors="coerce"
    ).fillna(0.0)

# --- Session state defaults ---
for k, v in {
    "analyzed": False,
    "data": None,
    "metrics": None,
    "insights": None,
    "inv_df": None,
    "slow_sku": None,
    "sheet_id": SHEET_ID_DEFAULT,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
st.session_state.setdefault("filter_channel", None)
CHANNEL_KEYWORDS = {
    "shopee": "Shopee",
    "lazada": "Lazada",
    "tiktok": "Tiktok",
    "facebook": "Facebook",
    "fb": "Facebook",
    "web": "Web",
}
# ----------------- SIDEBAR -----------------
sheet_id = st.sidebar.text_input("🔗 Google Sheet ID", st.session_state["sheet_id"], help="Đoạn giữa /d/ và /edit")
if sheet_id:
    st.session_state["sheet_id"] = sheet_id
    os.environ["SHEET_ID"] = sheet_id  # để utils.sheets sử dụng

# Memory (agent_state)
state = read_agent_state(sheet_id) if sheet_id else {}
dos_threshold = int(state.get("dos_threshold", 60))
discount_default = int(state.get("discount_default", 12))

dos_threshold = st.sidebar.number_input("Ngưỡng days_of_stock", 15, 180, dos_threshold, step=5)
discount_default = st.sidebar.number_input("Giảm giá mặc định (%)", 5, 50, discount_default, step=1)

col_mem1, col_mem2 = st.sidebar.columns(2)
if col_mem1.button("💾 Lưu cấu hình agent"):
    msg = write_agent_state(sheet_id, {"dos_threshold": dos_threshold, "discount_default": discount_default})
    st.sidebar.success(msg)
if col_mem2.button("🔄 Nạp cấu hình"):
    st.sidebar.success("Đã nạp lại agent_state.")

# Intent (lệnh tự nhiên)
st.sidebar.markdown("---")
user_cmd = st.sidebar.text_input("🗣️ Ra lệnh cho agent", placeholder="vd: tạo promo plan tuần này / xuất báo cáo")
run_cmd = st.sidebar.button("▶️ Thực thi lệnh")

# Nút chính
run_analyze = st.sidebar.button("🚀 Phân tích dữ liệu")

# ----------------- HÀM PHÂN TÍCH & LƯU STATE -----------------
def run_analysis_and_store(channel: str | None = None):
    sheet_id = st.session_state["sheet_id"]
    # 1) read
    data = {
        "orders": read_sheet("orders"),
        "inventory": read_sheet("inventory"),
        "customers": read_sheet("customers"),
    }
    # Lọc theo kênh nếu có
    if channel and ("orders" in data) and (data["orders"] is not None) and (not data["orders"].empty):
        if "channel" in data["orders"].columns:
             data["orders"] = data["orders"][data["orders"]["channel"].astype(str).str.lower() == channel.lower()]
        st.session_state["filter_channel"] = channel
    else:
        st.session_state["filter_channel"] = None  
    # 2) metrics + fallback từ orders (để luôn có total_revenue/…)
    metrics = compute_metrics(data.get("orders", pd.DataFrame()), data.get("inventory", pd.DataFrame()))

    orders_df = data.get("orders", pd.DataFrame()).copy()
    def _num(x):
        return pd.to_numeric(
            pd.Series(x, dtype="object").astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.replace(".", "", regex=False),
            errors="coerce"
        ).fillna(0.0)

    if not orders_df.empty:
        orders_df["date"] = pd.to_datetime(orders_df["date"], errors="coerce")
        orders_df["qty"] = _num(orders_df["qty"])
        orders_df["unit_price"] = _num(orders_df["unit_price"])
        orders_df = orders_df.dropna(subset=["date"])
        orders_df["revenue"] = orders_df["qty"] * orders_df["unit_price"]

        last_date = orders_df["date"].max()
        rev_7  = orders_df[orders_df["date"] >= last_date - pd.Timedelta(days=6)]["revenue"].sum()
        rev_28 = orders_df[orders_df["date"] >= last_date - pd.Timedelta(days=27)]["revenue"].sum()
        prev7  = orders_df[(orders_df["date"] >= last_date - pd.Timedelta(days=13)) &
                           (orders_df["date"] <= last_date - pd.Timedelta(days=7))]["revenue"].sum()
        prev28 = orders_df[(orders_df["date"] >= last_date - pd.Timedelta(days=55)) &
                           (orders_df["date"] <= last_date - pd.Timedelta(days=28))]["revenue"].sum()
        def pct(a, b): return float((a - b) / b * 100) if b and b != 0 else 0.0

        metrics.setdefault("total_revenue", float(orders_df["revenue"].sum()))
        metrics.setdefault("revenue_last_7d", float(rev_7))
        metrics.setdefault("revenue_last_28d", float(rev_28))
        metrics.setdefault("wow_change_pct", pct(rev_7, prev7))
        metrics.setdefault("mom_change_pct", pct(rev_28, prev28))
    else:
        metrics.setdefault("total_revenue", 0.0)
        metrics.setdefault("revenue_last_7d", 0.0)
        metrics.setdefault("revenue_last_28d", 0.0)
        metrics.setdefault("wow_change_pct", 0.0)
        metrics.setdefault("mom_change_pct", 0.0)
    st.session_state["orders_df"] = orders_df.copy() if 'orders_df' in locals() else pd.DataFrame()

    # 3) charts & insights
    try:
        insights = gen_insights(metrics)
    except Exception as e:
        insights = f"⚠️ LLM error: {e}"

    # 4) inventory: tính bù các cột cần cho promo_plan nếu thiếu
    inv_df = metrics.get("inventory", pd.DataFrame()).copy()
    need_cols = {"qty_7d","qty_30d","avg_daily_30d","days_of_stock","promo_suggest"}
    if not inv_df.empty and not need_cols.issubset(set(inv_df.columns)):

        # Làm sạch số
        for c in ["stock","cost","price"]:
            if c in inv_df.columns:
                inv_df[c] = to_number(inv_df[c])

        if not orders_df.empty:
            last_date = orders_df["date"].max()
            g7 = (orders_df[orders_df["date"] >= last_date - pd.Timedelta(days=6)]
      .groupby("sku", as_index=False)["qty"].sum()
      .rename(columns={"qty":"qty_7d"}))
            g30 = (orders_df[orders_df["date"] >= last_date - pd.Timedelta(days=29)]
       .groupby("sku", as_index=False)["qty"].sum()
       .rename(columns={"qty":"qty_30d"}))

        else:
            g7  = pd.DataFrame(columns=["sku","qty_7d"])
            g30 = pd.DataFrame(columns=["sku","qty_14d"])

        inv_df = (inv_df.merge(g7, on="sku", how="left")
                        .merge(g30, on="sku", how="left"))
        inv_df["qty_7d"] = inv_df["qty_7d"].fillna(0.0)
        inv_df["qty_30d"] = inv_df["qty_30d"].fillna(0.0)
        inv_df["avg_daily_30"] = (inv_df["qty_30d"] / 14).replace([float("inf")], 0).round(1)
        inv_df["days_of_stock"] = inv_df.apply(
            lambda r: (r["stock"]/r["avg_daily_30"]) if r["avg_daily_30"]>0 else np.nan, axis=1
        )
        inv_df["days_of_stock"] = inv_df["days_of_stock"].round(0).astype("Int64")

        def suggest(dos):
            if dos > 90: return f"{max(discount_default, 12)}% + bundle"
            if dos > 60: return f"{max(discount_default, 10)}%"
            if dos > 30: return f"{discount_default}%"
            return ""
        inv_df["promo_suggest"] = inv_df["days_of_stock"].apply(suggest)

        metrics["inventory"] = inv_df  # cập nhật để phần sau dùng chung

    # 5) lọc slow_sku theo ngưỡng
    slow_sku = None
    if not inv_df.empty and "days_of_stock" in inv_df.columns:
        cols_pref = ["sku","name","stock","qty_7d","qty_30d","avg_daily_30d","days_of_stock","promo_suggest"]
        cols_exist = [c for c in cols_pref if c in inv_df.columns]
        slow_sku = inv_df[inv_df["days_of_stock"] > dos_threshold][cols_exist].copy()

    # 6) Lưu state
    st.session_state["analyzed"] = True
    st.session_state["data"] = data
    st.session_state["metrics"] = metrics
    st.session_state["insights"] = insights
    st.session_state["inv_df"] = inv_df
    st.session_state["slow_sku"] = slow_sku
    st.session_state["orders_df"] = orders_df

    return metrics, insights, inv_df, slow_sku

# ----------------- PHÂN TÍCH DỮ LIỆU -----------------
metrics = insights = inv_df = slow_sku = None

if run_analyze:
    with st.spinner("⏳ Đang phân tích dữ liệu..."):
        metrics, insights, inv_df, slow_sku = run_analysis_and_store()

elif run_cmd and not st.session_state["analyzed"]:
    # nếu user bấm lệnh mà chưa phân tích lần nào
    with st.spinner("⏳ Chưa có dữ liệu, đang phân tích trước khi chạy lệnh..."):
        metrics, insights, inv_df, slow_sku = run_analysis_and_store()

elif st.session_state["analyzed"]:
    # dùng dữ liệu đã có trong state (tránh mất khi rerun)
    sheet_id = st.session_state["sheet_id"]
    metrics = st.session_state["metrics"]
    insights = st.session_state["insights"] or gen_insights(metrics)
    inv_df = st.session_state["inv_df"]
    slow_sku = st.session_state["slow_sku"]
else:
    st.info("👈 Nhập Google Sheet ID rồi bấm **Phân tích dữ liệu**. Bạn cũng có thể nhập **lệnh tự nhiên** cho agent ở sidebar.")
    st.stop()

# 3) Biểu đồ
st.subheader("📊 Biểu đồ doanh thu & Top SKU")
col1, col2 = st.columns(2)

# dùng orders_df mà bạn đã tính ở trên (đã có date, qty, unit_price)
orders_df = st.session_state.get("orders_df", pd.DataFrame())
df = orders_df.copy()

if df is None or df.empty:
    with col1: st.info("Chưa có dữ liệu đơn hàng để vẽ biểu đồ.")
    with col2: st.empty()
else:
    # đảm bảo kiểu dữ liệu
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"])
    if "revenue" not in df.columns:
        df["revenue"] = df["qty"] * df["unit_price"]

    # 1) Doanh thu theo ngày (line) — nhỏ
c1, c2 = st.columns(2)

# ---------- Chart 1: Doanh thu theo ngày ----------
daily = (
    df.groupby("date", as_index=False)["revenue"]
      .sum()
      .sort_values("date")
)
max_rev = float(daily["revenue"].max())
use_millions = max_rev >= 10_000_000  # ngưỡng hiển thị M

fig1, ax1 = plt.subplots(figsize=(5.2, 3), dpi=220)
ax1.plot(daily["date"], daily["revenue"], marker="o", markersize=3.2, linewidth=1.4)
ax1.set_title("Doanh thu theo ngày")
ax1.set_xlabel("Ngày")
ax1.set_ylabel("Doanh thu" + (" (triệu ₫)" if use_millions else " (₫)"))

# X-axis: locator + formatter thông minh
num_pts = len(daily)
interval = max(1, int(np.ceil(num_pts / 8)))  # ~8 mốc trên trục
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

for t in ax1.get_xticklabels():
    t.set_rotation(45)
    t.set_ha("right")

# để nhãn không bị cắt
fig1.tight_layout()

# Y-axis: tiền Việt + lưới
if use_millions:
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1_000_000:,.1f}M"))
else:
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f} ₫"))
ax1.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax1.grid(True, which="major", linestyle="--", alpha=0.35)
ax1.grid(True, which="minor", linestyle=":", alpha=0.18, axis="y")
ax1.margins(x=0.01)
fig1.tight_layout()

with c1:
    st.pyplot(fig1, clear_figure=True)
    if use_millions:
        st.caption("Ghi chú: 1M = 1.000.000₫")
# --- map tên sản phẩm từ inventory vào orders df ---
inv = st.session_state.get("inv_df", pd.DataFrame())
if inv is None or inv.empty:
    try:
        inv = read_sheet("inventory")  # fallback nếu chưa có trong state
    except Exception:
        inv = pd.DataFrame()

name_map = None
if not inv.empty and "sku" in inv.columns:
    name_col_inv = "product_name" if "product_name" in inv.columns else ("name" if "name" in inv.columns else None)
    if name_col_inv:
        name_map = (inv[["sku", name_col_inv]]
                    .dropna()
                    .astype({"sku":"string", name_col_inv:"string"})
                    .drop_duplicates("sku")
                    .set_index("sku")[name_col_inv])

# gắn tên vào df orders
if name_map is not None:
    df["item_name"] = df["sku"].astype("string").map(name_map)
else:
    df["item_name"] = pd.NA
# ---------- Chart 2: Top 10 sản phẩm theo doanh thu (ưu tiên tên) ----------
# nhóm theo SKU để chính xác số, rồi map tên
sku_rev = (df.groupby("sku", as_index=False)["revenue"]
             .sum()
             .sort_values("revenue", ascending=False)
             .head(10))

# lấy tên đã map sẵn
label_df = sku_rev.merge(
    df[["sku","item_name"]].dropna().drop_duplicates("sku"),
    on="sku", how="left"
)

def short(s, n=42):
    s = str(s) if pd.notna(s) else ""
    return (s[:n-1]+"…") if len(s) > n else s

# chỉ dùng TÊN; nếu thiếu tên thì fallback về SKU để tránh trống
label_df["label"] = label_df["item_name"].where(
    label_df["item_name"].notna() & (label_df["item_name"].astype(str).str.strip()!=""),
    label_df["sku"]
).map(short)

fig2, ax2 = plt.subplots(figsize=(5.2, 3), dpi=220)
ax2.barh(label_df["label"], label_df["revenue"])
ax2.invert_yaxis()
ax2.set_title("Top 10 sản phẩm theo doanh thu" + (" (đơn vị: M)" if use_millions else ""))
ax2.set_xlabel("Doanh thu" + (" (M)" if use_millions else " (₫)"))
ax2.set_ylabel("Sản phẩm")
ax2.xaxis.set_major_formatter(
    mtick.FuncFormatter((lambda x, _: f"{x/1_000_000:,.0f} M") if use_millions else (lambda x, _: f"{x:,.0f} ₫"))
)
ax2.grid(True, linestyle="--", alpha=0.3, axis="x")
fig2.tight_layout()
with c2:
    st.pyplot(fig2, clear_figure=True)

# 4) SKU chậm bán
st.subheader(f"🐢 SKU chậm bán (days_of_stock > {dos_threshold})")
if slow_sku is not None and not slow_sku.empty:
    st.dataframe(slow_sku, use_container_width=True)
else:
    st.info("Không có SKU chậm bán theo ngưỡng hiện tại hoặc thiếu dữ liệu.")

# 5) Nhận xét & gợi ý
st.subheader("💬 Nhận xét & Gợi ý từ AI")
parts = clean_bullets(insights)
if parts:
    if len(parts) >= 1:
        st.markdown("### 1️⃣ Phân tích tổng quan")
        st.markdown(parts[0], unsafe_allow_html=True)
    if len(parts) >= 2:
        st.markdown("### 2️⃣ Insight chính")
        st.markdown(parts[1], unsafe_allow_html=True)
    if len(parts) >= 3:
        st.markdown("### 3️⃣ Gợi ý khuyến mãi")
        st.markdown(parts[2], unsafe_allow_html=True)
else:
    st.write(insights)

st.success("✅ Hoàn tất phân tích!")

# 6) Hành động của agent (ACT)
st.markdown("---")
st.markdown("## 🛠️ Hành động của Agent")

# Chọn khung thời gian (chỉ 7 hoặc 30 ngày)
cfg_col, _ = st.columns([1, 3])
with cfg_col:
    plan_window_label = st.selectbox(
        "Khung thời gian Promo Plan",
        ["1 tuần (7 ngày)", "1 tháng (30 ngày)"],
        index=0
    )
window_days = 7 if "7" in plan_window_label else 30

colA, colB, colC = st.columns(3)

# 6.1 Ghi Promo Plan — đọc từ state để không mất khi rerun
with colA:
    if st.button(f"📄 Tạo & ghi Promo Plan ({window_days} ngày) vào Google Sheets",
                 use_container_width=True):
        try:
            slow = st.session_state.get("slow_sku")
            sid  = st.session_state.get("sheet_id")
            if slow is None or slow.empty:
                st.warning("Không có SKU chậm bán để ghi.")
            else:
                export_df = slow.copy()

                # lấy đúng cột theo lựa chọn 7d/30d
                export_df["qty_recent"] = np.where(
                    window_days == 7,
                    export_df.get("qty_7d", 0),
                    export_df.get("qty_30d", 0)
                )

                # tính lại avg_daily & days_of_stock theo lựa chọn
                export_df["avg_daily"] = (
                    export_df["qty_recent"] / window_days
                ).replace([np.inf], 0).round(1)

                export_df["days_of_stock"] = export_df.apply(
                    lambda r: (r["stock"] / r["avg_daily"]) if r["avg_daily"] > 0 else np.nan,
                    axis=1
                ).round(0).astype("Int64")

                # chỉ giữ cột cần thiết và đặt tên theo N ngày
                keep = ["sku","name","stock","qty_recent","avg_daily","days_of_stock","promo_suggest"]
                export_df = export_df[[c for c in keep if c in export_df.columns]]
                export_df = export_df.rename(columns={"qty_recent": f"qty_{window_days}d"})

                msg = write_promo_plan_to_sheet(sid, export_df)
                st.success(msg)
        except Exception as e:
            st.error(f"Lỗi ghi promo_plan: {e}")

# 6.2 Tải báo cáo/promo_plan
total = metrics.get("total_revenue", 0.0)
r7    = metrics.get("revenue_last_7d", 0.0)
r28   = metrics.get("revenue_last_28d", 0.0)
wow   = metrics.get("wow_change_pct", 0.0)
mom   = metrics.get("mom_change_pct", 0.0)

report_md = f"""# Báo cáo tuần
- Tổng doanh thu: {total:,.0f}₫
- 7 ngày: {r7:,.0f}₫ | 28 ngày: {r28:,.0f}₫
- WoW: {wow:.1f}% | MoM: {mom:.1f}%

## Gợi ý
{parts[2] if len(parts) >= 3 else "- (xem bảng tồn chậm bên trên)"}
"""
with colB:
    st.download_button("📥 Tải báo cáo tuần (Markdown)",
                       data=BytesIO(report_md.encode("utf-8")),
                       file_name="weekly_report.md",
                       mime="text/markdown")
with colC:
    if slow_sku is not None and not slow_sku.empty:
        st.download_button("📥 Tải promo_plan.csv",
                           data=slow_sku.to_csv(index=False).encode("utf-8"),
                           file_name="promo_plan.csv",
                           mime="text/csv")
    else:
        st.caption("Chưa có promo_plan để tải.")

# 7) Intent routing (rule-based đơn giản)
if run_cmd:
    st.markdown("---")
    st.markdown("## 🧭 Thực thi lệnh tự nhiên")
    text = (user_cmd or "").lower().strip()
    if not text:
        st.warning("Nhập lệnh trước khi chạy.")
    else:
        # 1) Lọc theo kênh (Shopee/Lazada/Web/Facebook/Tiktok…)
        chan = None
        for k, v in CHANNEL_KEYWORDS.items():
            if k in text:
                chan = v
                break

        if chan:
            with st.spinner(f"🔎 Đang phân tích riêng cho kênh **{chan}**..."):
                metrics, insights, inv_df, slow_sku = run_analysis_and_store(channel=chan)
            st.success(f"Đã lọc theo kênh **{chan}**. Biểu đồ & bảng đã cập nhật.")
            st.rerun()   # dừng ở đây để UI hiển thị theo kênh ngay
        # 2) Tạo promo plan
        if any(k in text for k in ["promo", "khuyến mãi", "tồn chậm", "plan"]):
            slow = st.session_state.get("slow_sku")
            sid = st.session_state.get("sheet_id")
            if slow is None or slow.empty:
                st.warning("Không có SKU chậm bán để tạo plan.")
            else:
                try:
                    msg = write_promo_plan_to_sheet(sid, slow)
                    st.success(f"Intent PROMO_PLAN: {msg}")
                except Exception as e:
                    st.error(f"Intent PROMO_PLAN lỗi: {e}")
        # 3) Báo cáo
        elif any(k in text for k in ["báo cáo", "report", "pdf", "doc"]):
            st.success("Intent REPORT: đã có nút tải báo cáo bên trên. (Có thể nâng cấp sang Google Docs sau.)")
        elif any(k in text for k in ["kênh", "channel", "shopee", "web", "lazada", "facebook", "tiktok"]):
            st.info("Intent ANALYZE_CHANNEL: thêm lọc kênh ở sidebar hoặc tự động lọc theo từ khoá (nâng cấp).")
        else:
            st.warning("Intent UNKNOWN: thử 'tạo promo plan', 'xuất báo cáo tuần', 'phân tích kênh Shopee'.")
