import os
import httpx

def gen_insights(metrics: dict):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Missing GEMINI_API_KEY in .env"

    model = "gemini-2.0-flash"  # hoặc "gemini-2.5-flash" nếu tài khoản có quyền
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

    prompt = (
        "Phân tích dữ liệu bán hàng sau và tạo **báo cáo dạng dễ đọc**, chia 3 phần rõ ràng:\n"
        "1️⃣ Phân tích tổng quan\n"
        "2️⃣ Insight chính\n"
        "3️⃣ Gợi ý khuyến mãi\n\n"
        "Hãy dùng emoji, chia dòng ngắn, thêm tiêu đề phụ, trình bày gọn gàng như ví dụ sau:\n\n"
        "🧾 BÁO CÁO PHÂN TÍCH DỮ LIỆU BÁN HÀNG\n"
        "1️⃣ Phân tích dữ liệu tổng quan\n"
        "- Tổng doanh thu: ...\n"
        "- Xu hướng: ...\n\n"
        "2️⃣ Insight chính\n"
        "💡 1. ...\n💡 2. ...\n\n"
        "3️⃣ Gợi ý khuyến mãi\n"
        "🎯 1. ...\n🎯 2. ...\n\n"
        "Hãy trình bày thật sạch đẹp, dễ đọc, không dùng Markdown phức tạp.\n\n"
        f"Dữ liệu đầu vào:\n{metrics}"
    )

    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=90)
        data = resp.json()

        if "candidates" not in data:
            err = data.get("error", {}).get("message", "Unknown LLM error")
            return f"⚠️ Gemini API error: {err}"

        text = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        if not text:
            return "⚠️ Gemini API returned empty response"

        cleaned = (
            text.replace("##", "")
            .replace("**", "")
            .replace("* ", "")
            .replace("\n\n\n", "\n\n")
            .strip()
        )
        cleaned = cleaned.replace("1️⃣", "\n\n1️⃣").replace("2️⃣", "\n\n2️⃣").replace("3️⃣", "\n\n3️⃣")

        formatted = f"🧾 BÁO CÁO PHÂN TÍCH DỮ LIỆU BÁN HÀNG\n\n{cleaned}"
        return formatted

    except Exception as e:
        return f"⚠️ Request failed: {e}"
