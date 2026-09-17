import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Pagina configuratie
st.set_page_config(page_title="Multi-Stock AI & Technical Scanner", layout="wide")
st.title("📈 Multi-Stock AI & Technical Scanner")

# User Input: Meerdere tickers gescheiden door komma's
default_tickers = "NVDA, AAPL, MSFT, TSLA, ASML.AS"
tickers_input = st.text_input("Voer aandelen in (gescheiden door komma's):", default_tickers)

# Tickers omzetten naar een lijst
ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

def calculate_composite_score(vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float):
    """
    Berekent de Totaal Score (1-10) op basis van 4 gewogen pijlers:
    1. Volume Breakout (25%)
    2. Opties / PCR (15%)
    3. Sentiment (20%)
    4. Technische Indicatoren (40%)
    """
    # 1. Volume Breakout Score (25%)
    if vol_ratio >= 2.0:
        vol_score = 10.0
    elif vol_ratio >= 1.5:
        vol_score = 8.5
    elif vol_ratio >= 1.2:
        vol_score = 7.0
    elif vol_ratio >= 1.0:
        vol_score = 5.5
    else:
        vol_score = 3.0

    # 2. Opties / PCR Score (15%)
    if put_call_ratio is not None:
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
    if short_float is not None:
        if short_float < 0.05:
            sentiment_score += 2.5
        elif short_float > 0.15:
            sentiment_score -= 2.0
    if vol_ratio > 1.3:
        sentiment_score += 1.5
    sentiment_score = min(10.0, max(1.0, sentiment_score))

    # 4. Technische Indicatoren Score (40%)
    tech_score = 5.0
    if 55 <= rsi_val <= 70:
        tech_score += 1.5
    elif rsi_val > 70:
        tech_score -= 1.0
    elif rsi_val < 30:
        tech_score += 1.0

    if macd_val > 0 and macd_val > macd_prev:
        tech_score += 2.0
    elif macd_val > 0:
        tech_score += 1.0

    if stoch_k > stoch_d:
        tech_score += 1.5

    tech_score = min(10.0, max(1.0, tech_score))

    # Totaal Score Berekening
    total_score = (
        (vol_score * 0.25) +
        (pcr_score * 0.15) +
        (sentiment_score * 0.20) +
        (tech_score * 0.40)
    )

    return round(total_score, 1), round(vol_score, 1), round(pcr_score, 1), round(sentiment_score, 1), round(tech_score, 1)


@st.cache_data(ttl=300)
def analyze_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", interval="1d")
        df_1h = stock.history(period="1mo", interval="1h")
        info = stock.info

        if df.empty or len(df) < 30:
            return None

        # 1. Volume Ratio (vs 20-daags MA)
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

        # Hourly Stochastic
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

        # 7. AI Model Simulation Scores
        mom_score = 5.0
        if macd_val > 0: mom_score += 1.5
        if stoch_k > stoch_d: mom_score += 1.5
        if rsi_val > 55: mom_score += 1.0
        if vol_ratio > 1.2: mom_score += 1.0
        mom_score = min(10.0, max(1.0, mom_score))

        ensemble_score = round((mom_score * 0.6) + (10 - (stoch_d * 0.05)) * 0.4, 1)

        # 8. Totaal Score Berekening
        totaal_score, vol_score, pcr_score, sent_score, tech_score = calculate_composite_score(
            vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float
        )

        return {
            "Ticker": symbol,
            "Koers": f"${c1:.2f}",
            "Totaal Score": totaal_score,
            "AI Ensemble": ensemble_score,
            "AI Momentum": round(mom_score, 1),
            "Volume Ratio": f"{vol_ratio:.2f}x",
            "RSI (14)": round(rsi_val, 1),
            "RSI Raw": rsi_val,
            "RSI Prev": rsi_prev,
            "MACD Status": macd_status,
            "Put/Call": f"{put_call_ratio:.2f}" if put_call_ratio else "N/B",
            "PCR Raw": put_call_ratio,
            "Short Float": f"{short_float * 100:.1f}%" if short_float else "N/B",
            "Stoch %K": round(stoch_k, 1),
            "Stoch %D": round(stoch_d, 1),
            "Stoch K Prev": stoch_k_prev,
            "Stoch 1H": stoch_1h_trend,
            "Stoch 1D": stoch_1d_trend,
            "3D Candles": candle_signal,
            "Tech Score": tech_score,
            "Volume Score": vol_score,
            "PCR Score": pcr_score,
            "Sentiment Score": sent_score
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

        # Sorteren op de Totaal Score (hoogste bovenaan)
        df_results = df_results.sort_values(by="Totaal Score", ascending=False)

        st.subheader("📊 Multi-Stock Scan Overzicht")
        
        # Weergave tabel voorbereiden
        display_cols = [
            "Ticker", "Koers", "Totaal Score", "AI Ensemble", "AI Momentum", 
            "Volume Ratio", "RSI (14)", "MACD Status", "Put/Call", 
            "Short Float", "Stoch 1H", "Stoch 1D", "3D Candles"
        ]

        st.dataframe(
            df_results[display_cols],
            column_config={
                "Totaal Score": st.column_config.NumberColumn(format="%.1f 🏆"),
                "AI Ensemble": st.column_config.NumberColumn(format="%.1f ⭐"),
                "AI Momentum": st.column_config.NumberColumn(format="%.1f 🔥"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        # Inzoomen op 1 specifiek aandeel voor uitgebreide weergave
        selected_ticker = st.selectbox("Selecteer een aandeel voor gedetailleerde kaartweergave:", df_results["Ticker"])
        selected_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

        if selected_data:
            st.write(f"### 🔍 Gedetailleerde Analyse voor {selected_ticker}")
            
            # Top Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Koers", selected_data["Koers"])
            m2.metric("Totaal Score", f"{selected_data['Totaal Score']} / 10")
            m3.metric("AI Ensemble", f"{selected_data['AI Ensemble']} / 10")
            m4.metric("Volume Ratio", selected_data["Volume Ratio"])
            m5.metric("RSI (14)", selected_data["RSI (14)"])

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                st.write("#### 📊 Technische Indicatoren")

                # RSI Custom Box
                rsi_val = selected_data["RSI Raw"]
                rsi_prev = selected_data["RSI Prev"]
                rsi_bg = "white"
                rsi_color = "black"
                if rsi_val > 70 and rsi_val < rsi_prev:
                    rsi_bg = "#ffcccc" # Rood (Overbought & daalt)
                    rsi_color = "#990000"
                elif rsi_val > 55:
                    rsi_bg = "#d4edda" # Groen
                    rsi_color = "#155724"

                st.markdown(f"""
                <div style="background-color:{rsi_bg}; color:{rsi_color}; padding:10px; border-radius:5px; margin-bottom:10px;">
                    <strong>RSI (14):</strong> {rsi_val:.2f}
                </div>
                """, unsafe_allow_html=True)

                # Stochastic Custom Box
                stoch_k = selected_data["Stoch %K"]
                stoch_d = selected_data["Stoch %D"]
                stoch_k_prev = selected_data["Stoch K Prev"]

                stoch_bg = "#d4edda" if (stoch_k > stoch_d and stoch_k > stoch_k_prev) else "#f8d7da"
                st.markdown(f"""
                <div style="background-color:{stoch_bg}; padding:10px; border-radius:5px; margin-bottom:10px;">
                    <strong>Stochastic (%K / %D):</strong> {stoch_k:.1f} / {stoch_d:.1f}
                </div>
                """, unsafe_allow_html=True)

                st.info(f"**MACD Status:** {selected_data['MACD Status']}")

            with col_right:
                st.write("#### 🔍 Sentiment & Trends")

                # Put/Call Status
                pcr_val = selected_data["PCR Raw"]
                if pcr_val:
                    pc_status = "Bullish (< 0.8) 🟢" if pcr_val < 0.8 else ("Bearish (> 1.0) 🔴" if pcr_val > 1.0 else "Neutraal 🟡")
                    st.write(f"**Put/Call Ratio:** {pcr_val:.2f} → *{pc_status}*")
                else:
                    st.write("**Put/Call Ratio:** Niet beschikbaar")

                st.write(f"**Short Float:** {selected_data['Short Float']}")
                st.write(f"**Stochastic Trend 1 Uur:** {selected_data['Stoch 1H']}")
                st.write(f"**Stochastic Trend 1 Dag:** {selected_data['Stoch 1D']}")
                st.write(f"**3-Daagse Candle Analyse:** {selected_data['3D Candles']}")

            st.write("---")
            st.write("#### 🎯 Score Opbouw Breakdown")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Technische Score (40%)", f"{selected_data['Tech Score']} / 10")
            sc2.metric("Volume Breakout Score (25%)", f"{selected_data['Volume Score']} / 10")
            sc3.metric("Sentiment Score (20%)", f"{selected_data['Sentiment Score']} / 10")
            sc4.metric("Opties/PCR Score (15%)", f"{selected_data['PCR Score']} / 10")
