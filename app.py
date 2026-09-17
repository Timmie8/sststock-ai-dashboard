import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Pagina configuratie
st.set_page_config(page_title="Multi-Stock AI Scanner", layout="wide")
st.title("📈 Multi-Stock AI & Technical Scanner")

# User Input: Meerdere tickers gescheiden door komma's
default_tickers = "NVDA, AAPL, MSFT, TSLA, ASML.AS"
tickers_input = st.text_input("Voer aandelen in (gescheiden door komma's):", default_tickers)

# Tickers omzetten naar een lijst
ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

@st.cache_data(ttl=300)
def analyze_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1d")
        df_1h = stock.history(period="1mo", interval="1h")
        info = stock.info

        if df.empty or len(df) < 30:
            return None

        # 1. Volume Ratio
        avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        vol_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # 2. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        rsi_val = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-2]

        # 3. Put/Call Ratio & Short Float
        put_call_ratio = info.get('putCallRatio', None)
        short_float = info.get('shortPercentOfFloat', None)

        # 4. Stochastic Oscillator (Daily & Hourly)
        def calc_stoch(data, k_period=14, d_period=3):
            low_min = data['Low'].rolling(window=k_period).min()
            high_max = data['High'].rolling(window=k_period).max()
            k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            return k, d

        df['Stoch_%K'], df['Stoch_%D'] = calc_stoch(df)
        stoch_k = df['Stoch_%K'].iloc[-1]
        stoch_d = df['Stoch_%D'].iloc[-1]
        stoch_k_prev = df['Stoch_%K'].iloc[-2]

        df_1h['Stoch_%K'], df_1h['Stoch_%D'] = calc_stoch(df_1h)
        stoch_1h_trend = "Stijgend 🟢" if df_1h['Stoch_%K'].iloc[-1] > df_1h['Stoch_%K'].iloc[-2] else "Dalend 🔴"
        stoch_1d_trend = "Stijgend 🟢" if stoch_k > stoch_k_prev else "Dalend 🔴"

        # 5. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        macd_val = df['MACD'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2]

        if macd_val > 0 and macd_val > macd_prev:
            macd_status = "Strong Buy 🚀"
        elif macd_val > 0:
            macd_status = "Bullish 🟢"
        else:
            macd_status = "Bearish 🔴"

        # 6. 3-Day Candlestick Analysis
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

        # 7. AI Scores Calculation
        mom_score = 5.0
        if macd_val > 0: mom_score += 1.5
        if stoch_k > stoch_d: mom_score += 1.5
        if rsi_val > 55: mom_score += 1.0
        if vol_ratio > 1.2: mom_score += 1.0
        mom_score = min(10.0, max(1.0, mom_score))

        ensemble_score = round((mom_score * 0.6) + (10 - (stoch_d * 0.05)) * 0.4, 1)

        return {
            "Ticker": symbol,
            "Koers": f"${c1:.2f}",
            "AI Ensemble": ensemble_score,
            "AI Momentum": mom_score,
            "Volume Ratio": f"{vol_ratio:.2f}x",
            "RSI (14)": round(rsi_val, 1),
            "MACD Status": macd_status,
            "Put/Call": f"{put_call_ratio:.2f}" if put_call_ratio else "N/B",
            "Short Float": f"{short_float * 100:.1f}%" if short_float else "N/B",
            "Stoch 1H": stoch_1h_trend,
            "Stoch 1D": stoch_1d_trend,
            "3D Candles": candle_signal
        }
    except Exception as e:
        return None

if ticker_list:
    results = []
    with st.spinner("Aandelen scannen..."):
        for t in ticker_list:
            res = analyze_stock(t)
            if res:
                results.append(res)

    if results:
        df_results = pd.DataFrame(results)

        st.subheader("📊 Scan Overzicht")
        
        # Tabel weergeven
        st.dataframe(
            df_results,
            column_config={
                "AI Ensemble": st.column_config.NumberColumn(format="%.1f ⭐"),
                "AI Momentum": st.column_config.NumberColumn(format="%.1f 🔥"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        # Optie om in te zoomen op 1 specifiek aandeel
        selected_ticker = st.selectbox("Selecteer een aandeel voor meer details:", df_results["Ticker"])
        selected_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

        if selected_data:
            st.write(f"### Details voor {selected_ticker}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Koers", selected_data["Koers"])
            col2.metric("AI Ensemble Score", f"{selected_data['AI Ensemble']} / 10")
            col3.metric("Volume Ratio", selected_data["Volume Ratio"])
            col4.metric("RSI", selected_data["RSI (14)"])
