import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 頁面配置
# ==========================================
st.set_page_config(page_title="CANSLIM 選股器 (修復版)", layout="wide", page_icon="📈")

st.title("📈 CANSLIM 策略選股器")
st.markdown("""
此工具篩選符合 **威廉·歐尼爾 (William O'Neil)** CANSLIM 成長股特徵的股票。
*數據來源: Yahoo Finance (免費數據，僅供參考)*
""")

# ==========================================
# 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ 篩選參數設定")

# [M] 市場趨勢
check_market = st.sidebar.checkbox("啟用 [M] 市場趨勢檢查 (SPY > 50MA)", value=True)

st.sidebar.subheader("基本面與技術面標準")
# [C] & [A] 盈餘與基本面
min_eps_growth = st.sidebar.slider("[C/A] 最低 EPS 成長率 (%, YoY)", 0, 100, 20, 5) / 100
min_roe = st.sidebar.slider("[A] 最低 ROE (%)", 0, 40, 15, 1) / 100

# [N] 股價位置
near_high_pct = st.sidebar.slider("[N] 距離 52 週新高範圍 (%)", 5, 50, 15, 5) / 100

# [L] 相對強度
rs_rank_threshold = st.sidebar.slider("[L] RS 排名門檻 (前 %)", 10, 100, 50, 10) / 100

# 掃描範圍
st.sidebar.subheader("掃描設定")
scan_scope_option = st.sidebar.selectbox(
    "掃描範圍 (股票數量)",
    options=["測試用 (前 20 檔)", "快速掃描 (前 50 檔)", "標準掃描 (前 100 檔)", "完整 S&P 500 (極慢)"],
    index=1,
    help="雲端免費資源有限，建議選擇前 50 檔以免超時。"
)

# 解析掃描範圍
scope_map = {
    "測試用 (前 20 檔)": 20,
    "快速掃描 (前 50 檔)": 50,
    "標準掃描 (前 100 檔)": 100,
    "完整 S&P 500 (極慢)": 505
}
scan_limit = scope_map[scan_scope_option]

# ==========================================
# 核心邏輯函數
# ==========================================

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """
    獲取 S&P 500 清單，包含失敗時的備用清單。
    使用 html5lib 避免雲端 lxml 安裝錯誤。
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # 關鍵修正：指定 flavor='html5lib'
        table = pd.read_html(url, flavor='html5lib')
        return table[0]['Symbol'].tolist()
    except Exception as e:
        st.warning(f"無法從維基百科抓取清單 (網路或解析錯誤)，改用內建熱門股清單。")
        # 備用清單 (市值前 50 大)
        return [
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
            'V', 'JPM', 'XOM', 'WMT', 'UNH', 'MA', 'PG', 'JNJ', 'ORCL', 'HD',
            'MRK', 'COST', 'ABBV', 'KO', 'BAC', 'PEP', 'CVX', 'CRM', 'NFLX', 'AMD',
            'QCOM', 'ADBE', 'TMO', 'LIN', 'ACN', 'MCD', 'DIS', 'ABT', 'CSCO', 'WFC',
            'INTC', 'CMCSA', 'INTU', 'VZ', 'AMAT', 'PFE', 'IBM', 'PM', 'CAT', 'NOW'
        ]

def check_market_trend():
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo") 
        if len(hist) < 50: return True, 0, 0
        
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        price = hist['Close'].iloc[-1]
        return price > ma50, price, ma50
    except:
        return True, 0, 0

def analyze_stock(ticker):
    """分析單一股票"""
    try:
        # 修正符號: 維基百科用 '.', Yahoo Finance 用 '-' (例如 BRK.B -> BRK-B)
        ticker = ticker.replace('.', '-')
        stock = yf.Ticker(ticker)
        
        # 1. 技術面數據
        hist = stock.history(period="1y")
        if len(hist) < 200: return None 
        
        current_price = hist['Close'].iloc[-1]
        high_52 = hist['High'].max()
        
        # [N] 檢查
        if current_price < high_52 * (1 - near_high_pct):
            return None

        # 2. 基本面數據
        info = stock.info
        
        # [A] ROE 檢查
        roe = info.get('returnOnEquity', None)
        if roe is None or roe < min_roe:
            return None
            
        # [C] 成長率檢查
        e_growth = info.get('earningsGrowth', None)
        # 寬容處理：如果沒有數據，暫不剔除，避免篩不出任何結果
        if e_growth is not None and e_growth < min_eps_growth:
            return None

        # [L] 計算 RS
        start_price = hist['Close'].iloc[0]
        rs_raw = (current_price - start_price) / start_price
        
        return {
            '代碼': ticker,
            '公司名稱': info.get('shortName', ticker),
            '現價': round(current_price, 2),
            'RS強度(1年漲幅%)': round(rs_raw * 100, 2),
            'EPS成長(預估%)': round(e_growth * 100, 2) if e_growth else 'N/A',
            'ROE(%)': round(roe * 100, 2) if roe else 'N/A',
            '離52週高點(%)': round((current_price/high_52 - 1) * 100, 2)
        }
    except Exception:
        return None

def plot_candlestick(ticker):
    """繪製 K 線圖"""
    try:
        data = yf.Ticker(ticker).history(period="1y")
        if data.empty: return None

        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=ticker
        )])

        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MA50'], 
            line=dict(color='orange', width=1.5), name='50日均線'
        ))

        fig.update_layout(
            title=f'<b>{ticker} 日 K 線圖</b>',
            yaxis_title='股價',
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    except:
        return None

# ==========================================
# 主介面邏輯
# ==========================================

# 1. 市場狀態
with st.container():
    st.subheader("1️⃣ 市場環境檢查")
    if check_market:
        is_bull, spy_price, spy_ma = check_market_trend()
        if is_bull:
            st.success(f"✅ 市場多頭 (SPY ${spy_price:.0f} > 50MA ${spy_ma:.0f})")
        else:
            st.error(f"🛑 市場空頭/震盪 (SPY ${spy_price:.0f} < 50MA ${spy_ma:.0f})")
    else:
        st.info("已略過市場檢查")

st.divider()

# 2. 篩選
st.subheader("2️⃣ 執行篩選")
run_button = st.button("🚀 開始掃描股票", type="primary", use_container_width=True)

if 'screener_results' not in st.session_state:
    st.session_state['screener_results'] = None

if run_button:
    tickers = get_sp500_tickers()
    # 限制數量
    target_list = tickers[:scan_limit]
    
    results = []
    my_bar = st.progress(0, text="準備開始...")
    status = st.empty()

    for i, ticker in enumerate(target_list):
        status.text(f"正在分析 ({i+1}/{len(target_list)}): {ticker}")
        data = analyze_stock(ticker)
        if data:
            results.append(data)
        my_bar.progress((i + 1) / len(target_list))
        
    my_bar.empty()
    status.empty()
    
    if results:
        df = pd.DataFrame(results)
        df['RS_Percentile'] = df['RS強度(1年漲幅%)'].rank(pct=True)
        # 過濾 RS
        df_final = df[df['RS_Percentile'] >= (1 - rs_rank_threshold)].sort_values(by='RS強度(1年漲幅%)', ascending=False)
        st.session_state['screener_results'] = df_final
        st.success(f"掃描完成！找到 {len(df_final)} 檔股票。")
    else:
        st.session_state['screener_results'] = pd.DataFrame()
        st.warning("無符合條件的結果，請放寬條件。")

# 3. 結果與圖表
if st.session_state['screener_results'] is not None and not st.session_state['screener_results'].empty:
    df_res = st.session_state['screener_results']
    
    st.divider()
    st.subheader("3️⃣ 篩選結果")
    st.dataframe(df_res.drop(columns=['RS_Percentile']), use_container_width=True)
    
    st.subheader("📊 K 線圖檢視")
    opts = [f"{r['代碼']} - {r['公司名稱']}" for _, r in df_res.iterrows()]
    sel = st.selectbox("選擇股票:", opts)
    
    if sel:
        ticker_sel = sel.split(" - ")[0]
        fig = plot_candlestick(ticker_sel)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
