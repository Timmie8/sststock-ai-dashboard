import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Streamlit Pagina Configuratie
st.set_page_config(
    page_title="AI Swingtrade Scanner (1-5 Dagen)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live AI Swingtrade Scanner (1–5 Dagen)")
st.caption("Live analyse op basis van Sentiment, Volume Breakouts, Opties (PCR), Money Flow, Accumulatie en Technische Indicatoren.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Instellingen & Watchlist")
user_input = st.sidebar.text_input(
    "Vul tickers in (gescheiden door komma's):",
    value="NVDA, TSLA, AMD, PLTR, AAPL"
)

tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
scan_button = st.sidebar.button("🚀 Start Live Scan", type="primary")


# --- UNIFORME AI SCORE BEREKENING (Gelijk aan het Multi-Stock model) ---
def calculate_composite_score(vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float, mfi_val, ad_trend_3d):
    """
    Gewogen Totaal Score (1-10):
    1. Volume & Money Flow (25%)
    2. Opties / PCR (15%)
    3. Sentiment (20%)
    4. Technische Indicatoren & Accumulatie (40%)
    """
    # 1. Volume Breakout & Money Flow Score (25%)
    vol_score = 5.0
    if vol_ratio >= 1.5:
        vol_score += 2.5
    elif vol_ratio >= 1.1:
        vol_score += 1.0

    if mfi_val >= 60:
        vol_score += 2.5
    elif mfi_val >= 45:
        vol_score += 1.0
    elif mfi_val < 35:
        vol_score -= 1.5
    vol_score = min(10.0, max(1.0, vol_score))

    # 2. Opties / PCR Score (15%)
    if put_call_ratio is not None and not np.isnan(put_call_ratio):
        if put_call_ratio < 0.8:
            pcr_score = 9.0  # Bullish
        elif put_call_ratio <= 1.0:
            pcr_score = 6.5  # Neutraal
        else:
            pcr_score = 3.0  # Bearish
    else:
        pcr_score = 5.0  # Fallback

    # 3. Sentiment Score (20%)
    sentiment_score = 5.0
    if short_float is not None and not np.isnan(short_float):
        if short_float < 0.05:
            sentiment_score += 2.5
        elif short_float > 0.15:
            sentiment_score -= 2.0
    if vol_ratio > 1.3:
        sentiment_score += 1.5
    sentiment_score = min(10.0, max(1.0, sentiment_score))

    # 4. Technische Indicatoren & Accumulatie Score (40%)
    tech_score = 5.0
    if 55 <= rsi_val <= 70:
        tech_score += 1.0
    elif rsi_val > 70:
        tech_score -= 1.0

    if macd_val > 0 and macd_val > macd_prev:
        tech_score += 1.5

    if stoch_k > stoch_d:
        tech_score += 1.0

    # 3-daagse Accumulatie / Distributie invloed
    if ad_trend_3d == "Accumulatie 🟢":
        tech_score += 1.5
    elif ad_trend_3d == "Distributie 🔴":
        tech_score -= 1.5

    tech_score = min(10.0, max(1.0, tech_score))

    # Totaal Score Berekening
    total_score = (
        (vol_score * 0.25) +
        (pcr_score * 0.15) +
        (sentiment_score * 0.20) +
        (tech_score * 0.40)
    )

    return round(total_score, 1)


# --- CORE ANALYSE FUNCTIE ---
def get_live_swing_data(symbol):
    ticker = yf.Ticker(symbol)
    
    # 1. Live/Daily Data
    df = ticker.history(period="60d", interval="1d")
    if df.empty or len(df) < 30:
        return None
        
    info = ticker.info
    live_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    day_change_pct = ((live_price - prev_close) / prev_close) * 100
    
    # 2. Volume Spike & Ratio
    current_volume = df['Volume'].iloc[-1]
    avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = current_volume / avg_vol_20d if avg_vol_20d > 0 else 1.0
    
    # 3. Money Flow Index (1D MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    pos_mf14 = pos_flow.rolling(14).sum()
    neg_mf14 = neg_flow.rolling(14).sum()
    mfi = 100 - (100 / (1 + (pos_mf14 / neg_mf14)))
    mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50.0

    if mfi_val >= 60:
        mfi_status = "Bullish 🟢"
    elif mfi_val <= 40:
        mfi_status = "Bearish 🔴"
    else:
        mfi_status = "Neutraal 🟡"

    # 4. 3-Daagse Accumulatie / Distributie (A/D)
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    clv = clv.fillna(0)
    ad_line = (clv * df['Volume']).cumsum()
    ad_diff_3d = ad_line.iloc[-1] - ad_line.iloc[-4]
    
    if ad_diff_3d > 0:
        ad_trend_3d = "Accumulatie 🟢"
    elif ad_diff_3d < 0:
        ad_trend_3d = "Distributie 🔴"
    else:
        ad_trend_3d = "Neutraal 🟡"

    # 5. Support & Resistance (Steun en Weerstand)
    high_20 = df['High'].iloc[-21:-1].max()
    low_20 = df['Low'].iloc[-21:-1].min()
    prev_high = df['High'].iloc[-2]
    prev_low = df['Low'].iloc[-2]
    pivot = (prev_high + prev_low + prev_close) / 3
    resistance_1 = (2 * pivot) - prev_low
    support_1 = (2 * pivot) - prev_high

    effective_resistance = min(high_20, resistance_1) if min(high_20, resistance_1) > live_price else max(high_20, resistance_1)
    effective_support = max(low_20, support_1) if max(low_20, support_1) < live_price else min(low_20, support_1)

    # 6. Technische Indicatoren (EMA, RSI, Stochastic, MACD)
    ema5 = ta.trend.ema_indicator(df['Close'], window=5).iloc[-1]
    ema15 = ta.trend.ema_indicator(df['Close'], window=15).iloc[-1]
    rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
    
    # Stochastic
    low_min14 = df['Low'].rolling(window=14).min()
    high_max14 = df['High'].rolling(window=14).max()
    stoch_k_series = 100 * ((df['Close'] - low_min14) / (high_max14 - low_min14))
    stoch_d_series = stoch_k_series.rolling(window=3).mean()
    stoch_k = stoch_k_series.iloc[-1]
    stoch_d = stoch_d_series.iloc[-1]

    # MACD
    macd = ta.trend.MACD(df['Close'])
    macd_val = macd.macd().iloc[-1]
    macd_prev = macd.macd().iloc[-2]

    # 7. Opties Put/Call Ratio
    pcr_volume = None
    pcr_status = "Geen Data"
    try:
        if ticker.options:
            nearest_exp = ticker.options[0]
            opt_chain = ticker.option_chain(nearest_exp)
            calls_vol = opt_chain.calls['volume'].sum()
            puts_vol = opt_chain.puts['volume'].sum()
            if calls_vol > 0:
                pcr_volume = puts_vol / calls_vol
                if pcr_volume < 0.8: pcr_status = "Bullish 🟢"
                elif pcr_volume > 1.0: pcr_status = "Bearish 🔴"
                else: pcr_status = "Neutraal 🟡"
    except Exception:
        pass

    # 8. Short Float & Sector
    short_pct = info.get('shortPercentOfFloat', 0) or 0
    sector = info.get('sector', 'Onbekend')

    # 9. AI Score Berekening (Geharmoniseerd)
    live_ai_score = calculate_composite_score(
        vol_ratio=vol_ratio,
        rsi_val=rsi,
        macd_val=macd_val,
        macd_prev=macd_prev,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        put_call_ratio=pcr_volume,
        short_float=short_pct,
        mfi_val=mfi_val,
        ad_trend_3d=ad_trend_3d
    )

    # 10. 3D Candlestick Signal
    c3, c2, c1 = df['Close'].iloc[-3], df['Close'].iloc[-2], df['Close'].iloc[-1]
    o3, o2, o1 = df['Open'].iloc[-3], df['Open'].iloc[-2], df['Open'].iloc[-1]
    bullish_days = sum([1 for o, c in [(o3,c3), (o2,c2), (o1,c1)] if c > o])
    total_return_3d = ((c1 - o3) / o3) * 100

    if bullish_days == 3 and total_return_3d > 2:
        candle_signal = f"Zeer Sterk 🚀 (+{total_return_3d:.1f}%)"
    elif bullish_days >= 2 and total_return_3d > 0:
        candle_signal = f"Matig Bullish 📈 (+{total_return_3d:.1f}%)"
    elif total_return_3d < 0:
        candle_signal = f"Bearish 📉 ({total_return_3d:.1f}%)"
    else:
        candle_signal = f"Neutraal ➖ ({total_return_3d:.1f}%)"

    # Trade Niveaus
    entry = round(live_price, 2)
    sl = round(live_price * 0.965, 2)
    tp1 = round(live_price * 1.045, 2)
    tp2 = round(live_price * 1.085, 2)

    return {
        "Ticker": symbol,
        "Sector": sector,
        "Koers": f"${entry}",
        "Verandering": f"{round(day_change_pct, 2)}%",
        "AI Score": live_ai_score,
        "Signaal": "BUY / LONG 🟢" if live_ai_score >= 7.0 else ("WATCH 🟠" if live_ai_score >= 5.0 else "AVOID / SHORT 🔴"),
        "1D MoneyFlow Status": mfi_status,
        "3D Acc/Dist": ad_trend_3d,
        "3D Candles": candle_signal,
        "Support": f"${effective_support:.2f}",
        "Resistance": f"${effective_resistance:.2f}",
        "Volume Ratio": f"{round(vol_ratio, 2)}x",
        "RSI": round(rsi, 1),
        "EMA Trend": "Bullish" if ema5 > ema15 else "Bearish",
        "Put/Call Ratio": f"{round(pcr_volume, 2)} ({pcr_status})" if pcr_volume else "N/B",
        "Short Float": f"{round(short_pct * 100, 1)}%",
        "Entry": f"${entry}",
        "Stop Loss (-3.5%)": f"${sl}",
        "TP1 (1-3d)": f"${tp1}",
        "TP2 (3-5d)": f"${tp2}"
    }

# --- HOOFDSCHERM LOGICA ---
if scan_button or tickers:
    st.write(f"### Analyseren van: {', '.join(tickers)}")
    results = []
    
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(tickers):
        data = get_live_swing_data(ticker)
        if data:
            results.append(data)
        progress_bar.progress((idx + 1) / len(tickers))
    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by="AI Score", ascending=False)
        
        st.subheader("📊 Ranking & AI Scores")

        # Hoofdtabel met alle gevraagde indicatoren & scores
        display_cols = [
            "Ticker", "AI Score", "Signaal", "Koers", "Verandering", 
            "1D MoneyFlow Status", "3D Acc/Dist", "3D Candles", 
            "Support", "Resistance", "Volume Ratio", "RSI", "Put/Call Ratio", "Short Float"
        ]

        st.dataframe(
            df_res[display_cols],
            column_config={
                "AI Score": st.column_config.NumberColumn(format="%.1f 🏆"),
                "1D MoneyFlow Status": st.column_config.TextColumn("1D Money Flow"),
                "3D Candles": st.column_config.TextColumn("3D Candles"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.subheader("🎯 Concrete Trade Setups (1-5 Dagen)")
        for item in results:
            with st.expander(f"{item['Ticker']} — AI Score: {item['AI Score']}/10 ({item['Signaal']})"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Entry Level", item["Entry"])
                col2.metric("Stop Loss", item["Stop Loss (-3.5%)"])
                col3.metric("Take Profit 1", item["TP1 (1-3d)"])
                col4.metric("Take Profit 2", item["TP2 (3-5d)"])
                
                st.write(f"**Support:** {item['Support']} | **Resistance:** {item['Resistance']}")
                st.write(f"**Sector:** {item['Sector']} | **EMA Trend:** {item['EMA Trend']} | **Volume Ratio:** {item['Volume Ratio']}")
