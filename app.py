import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go # 引入 Plotly 用於繪製 K 線圖
from datetime import datetime

# ==========================================
# 頁面配置
# ==========================================
st.set_page_config(page_title="CANSLIM 選股器 (含K線圖)", layout="wide", page_icon="📈")

st.title("📈 CANSLIM 策略選股器 (Web版)")
st.markdown("""
此工具篩選符合 **威廉·歐尼爾 (William O'Neil)** CANSLIM 成長股特徵的股票。
篩選出清單後，請利用下方的 **K 線圖** 功能檢查是否有合適的技術型態 (如杯柄型)。
*數據來源: Yahoo Finance (免費數據，僅供參考)*
""")

# ==========================================
# 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ 篩選參數設定")

# [M] 市場趨勢
check_market = st.sidebar.checkbox("啟用 [M] 市場趨勢檢查 (SPY > 50MA)", value=True, help="若 SPY 在 50 日均線下方，通常不建議進場做多。")

st.sidebar.subheader("基本面與技術面標準")
# [C] & [A] 盈餘與基本面
min_eps_growth = st.sidebar.slider("[C/A] 最低 EPS 成長率 (%, YoY)", 0, 100, 25, 5, help="最近一季或年度的 EPS 成長率。CANSLIM 標準通常要求 >25%。") / 100
min_roe = st.sidebar.slider("[A] 最低 ROE (%)", 0, 40, 15, 1, help="股東權益報酬率。標準通常要求 >17%。") / 100

# [N] 股價位置
near_high_pct = st.sidebar.slider("[N] 距離 52 週新高範圍 (%)", 5, 50, 15, 5, help="股價應接近一年來的高點，準備突破。") / 100

# [L] 相對強度
rs_rank_threshold = st.sidebar.slider("[L] RS 排名門檻 (前 %)", 10, 100, 50, 10, help="在篩選出的股票池中，只保留相對強度(過去一年漲幅)排名前多少%的股票。") / 100

# 掃描範圍
st.sidebar.subheader("掃描設定")
scan_scope_option = st.sidebar.selectbox(
    "掃描範圍 (股票數量)",
    options=["測試用 (前 20 檔)", "快速掃描 (前 50 檔)", "標準掃描 (前 100 檔)", "完整 S&P 500 (極慢)"],
    index=1,
    help="因 Yahoo Finance API 速度限制，掃描大量股票需要較長時間。"
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
@st.cache_data(ttl=3600) # 緩存 1 小時
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        return table[0]['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法獲取 S&P 500 清單: {e}")
        return ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META'] # 備用

def check_market_trend():
    try:
        spy = yf.Ticker("SPY")
        # 抓取足夠計算 50MA 的數據
        hist = spy.history(period="3mo") 
        if len(hist) < 50: return True, 0, 0 # 數據不足時默認通過
        
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        price = hist['Close'].iloc[-1]
        return price > ma50, price, ma50
    except:
        return True, 0, 0

def analyze_stock(ticker):
    """分析單一股票"""
    try:
        stock = yf.Ticker(ticker)
        
        # 1. 技術面數據 (快)
        hist = stock.history(period="1y")
        if len(hist) < 200: return None # 上市未滿一年
        
        current_price = hist['Close'].iloc[-1]
        high_52 = hist['High'].max()
        
        # [N] 檢查: 接近 52 週新高
        if current_price < high_52 * (1 - near_high_pct):
            return None

        # 2. 基本面數據 (慢，易失敗)
        info = stock.info
        
        # [A] ROE 檢查
        # yfinance 的 info 欄位經常變動或缺失，使用 get 方法並給予默認值
        roe = info.get('returnOnEquity', None)
        if roe is None or roe < min_roe:
            return None
            
        # [C] & [A] 成長率檢查
        # 使用 earningsGrowth (最近一年預估成長) 作為簡化代理
        e_growth = info.get('earningsGrowth', None)
        # 如果抓不到 growth 數據，暫時先放行，避免篩不出東西 (免費數據的局限)
        if e_growth is not None and e_growth < min_eps_growth:
            return None

        # [L] 計算 RS (原始 1 年漲幅)
        start_price = hist['Close'].iloc[0]
        if start_price <= 0: return None
        rs_raw = (current_price - start_price) / start_price
        
        return {
            '代碼': ticker,
            '公司名稱': info.get('shortName', 'N/A'),
            '現價': round(current_price, 2),
            'RS強度(1年漲幅%)': round(rs_raw * 100, 2),
            'EPS成長(預估%)': round(e_growth * 100, 2) if e_growth else 'N/A',
            'ROE(%)': round(roe * 100, 2) if roe else 'N/A',
            '離52週高點(%)': round((current_price/high_52 - 1) * 100, 2),
            '產業': info.get('industry', 'N/A')
        }
    except Exception as e:
        # print(f"Error analyzing {ticker}: {e}") # Debug 用
        return None

# ==========================================
# 繪圖函數 (K線圖)
# ==========================================
def plot_candlestick(ticker):
    try:
        # 抓取 1 年數據用於觀察型態
        data = yf.Ticker(ticker).history(period="1y")
        
        if data.empty:
            st.error(f"無法獲取 {ticker} 的歷史數據。")
            return None

        # 建立 K 線圖物件
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        )])

        # 增加 50 日均線 (輔助判斷趨勢)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA50'], 
            line=dict(color='orange', width=1.5), 
            name='50日均線'
        ))

        # 設定圖表版面
        fig.update_layout(
            title=f'<b>{ticker} 日 K 線圖 (過去一年)</b>',
            yaxis_title='股價',
            xaxis_rangeslider_visible=False, # 隱藏底部滑桿，讓圖更清爽
            template="plotly_dark", # 使用深色主題，看起來更專業
            height=600,
            hovermode='x unified', # 游標懸停時顯示資訊
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
        
    except Exception as e:
        st.error(f"繪圖失敗: {e}")
        return None

# ==========================================
# 主介面邏輯
# ==========================================

# 1. 市場狀態顯示區
with st.container():
    st.subheader("1️⃣ 市場環境檢查 (M)")
    if check_market:
        is_bull, spy_price, spy_ma = check_market_trend()
        if spy_price == 0:
            st.warning("⚠️ 無法獲取 SPY 數據，跳過市場檢查。")
        elif is_bull:
            st.success(f"✅ 市場狀態：**多頭趨勢** (SPY ${spy_price:.2f} > 50MA ${spy_ma:.2f})。適合進場尋找飆股。")
        else:
            st.error(f"🛑 市場狀態：**空頭或震盪** (SPY ${spy_price:.2f} < 50MA ${spy_ma:.2f})。")
            st.caption("根據 CANSLIM 原則，大盤不佳時應保守操作或空手。您仍可執行篩選以觀察強勢股，但需謹慎。")
    else:
        st.info("已略過市場趨勢檢查。")

st.divider()

# 2. 執行篩選區
st.subheader("2️⃣ 執行篩選")
col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("🚀 開始掃描股票", type="primary", use_container_width=True)
with col2:
    st.caption(f"當前設定將掃描 S&P 500 中的前 **{scan_limit}** 檔股票。請耐心等待。")

# 初始化 session state 來儲存結果，避免重新整理後消失
if 'screener_results' not in st.session_state:
    st.session_state['screener_results'] = None

if run_button:
    tickers = get_sp500_tickers()
    tickers = tickers[:scan_limit] # 限制數量
    
    results = []
    
    # 進度條顯示元件
    progress_text = "掃描進行中，請稍候..."
    my_bar = st.progress(0, text=progress_text)
    status_placeholder = st.empty()

    # 開始迴圈掃描
    for i, ticker in enumerate(tickers):
        status_placeholder.text(f"正在分析 ({i+1}/{len(tickers)}): {ticker} ...")
        data = analyze_stock(ticker)
        if data:
            results.append(data)
        # 更新進度條
        my_bar.progress((i + 1) / len(tickers), text=progress_text)
        
    # 清除進度顯示
    my_bar.empty()
    status_placeholder.empty()
    
    # 處理結果
    if results:
        df = pd.DataFrame(results)
        # [L] 相對強度排名過濾
        # 計算百分比排名 (數字越大越好)
        df['RS_Percentile'] = df['RS強度(1年漲幅%)'].rank(pct=True, ascending=True)
        # 過濾掉排名在門檻以下的
        df_final = df[df['RS_Percentile'] >= (1 - rs_rank_threshold)].sort_values(by='RS強度(1年漲幅%)', ascending=False)
        
        # 將最終結果存入 session state
        st.session_state['screener_results'] = df_final
        st.toast(f"掃描完成！共找到 {len(df_final)} 檔符合條件的股票。", icon="🎉")
    else:
        st.session_state['screener_results'] = pd.DataFrame() # 空的 DataFrame
        st.error("沒有股票符合當前嚴格的篩選條件，請嘗試放寬側邊欄的參數。")

# 3. 結果顯示與 K 線圖互動區
if st.session_state['screener_results'] is not None and not st.session_state['screener_results'].empty:
    df_result = st.session_state['screener_results']
    
    st.divider()
    st.subheader(f"3️⃣ 篩選結果 ({len(df_result)} 檔)")
    
    # 顯示資料表，針對 RS 強度做顏色漸層
    st.dataframe(
        df_result.drop(columns=['RS_Percentile']).style.background_gradient(subset=['RS強度(1年漲幅%)'], cmap='Greens'),
        use_container_width=True,
        height=300
    )
    
    st.divider()
    st.subheader("📊 個股 K 線圖型態檢視")
    st.info("👇 請從下方選單選擇一支股票，觀察其 K 線圖是否出現「杯柄型」或其他突破型態。")
    
    # 股票選擇下拉選單
    # 創建選項清單，格式為 "代碼 - 公司名稱"
    stock_options = [f"{row['代碼']} - {row['公司名稱']}" for index, row in df_result.iterrows()]
    
    col_select, col_chart_info = st.columns([1, 3])
    with col_select:
        selected_option = st.selectbox("選擇要查看的股票:", options=stock_options)
        selected_ticker = selected_option.split(" - ")[0]

    # 繪製 K 線圖
    if selected_ticker:
        with st.spinner(f"正在繪製 {selected_ticker} 的 K 線圖..."):
            fig = plot_candlestick(selected_ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("圖表說明：橘色線為 50 日移動平均線 (MA50)。CANSLIM 買點通常發生在股價突破盤整區間且站上 MA50 之時。")

elif st.session_state['screener_results'] is not None and st.session_state['screener_results'].empty:
    st.divider()
    st.warning("無符合條件的結果。請調整左側參數後重新掃描。")

else:
    # 尚未執行過掃描時的提示
    st.divider()
    st.info("👈 請在左側調整參數，並點擊上方「開始掃描股票」按鈕。")