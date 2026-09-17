import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Pagina configuratie
st.set_page_config(page_title="AI Stock Technical Dashboard", layout="wide")
st.title("📈 Swingtrade & AI Technical Dashboard")

# User Input
ticker_symbol = st.text_input("Voer Aandeel Ticker in (bijv. AAPL, NVDA, ASML.AS):", "NVDA").upper()

@st.cache_data(ttl=300)
def load_data(symbol):
    stock = yf.Ticker(symbol)
    # 1D data voor algemene indicatoren
    df_daily = stock.history(period="6mo", interval="1d")
    # 1H data voor uurs-stochastics
    df_hourly = stock.history(period="1mo", interval="1h")
    info = stock.info
    return df_daily, df_hourly, info

if ticker_symbol:
    try:
        df, df_1h, info = load_data(ticker_symbol)
        
        if df.empty:
            st.error("Geen data gevonden voor deze ticker.")
        else:
            # ==================== BEREKENINGEN ====================
            
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
            
            # Hourly Stochastic
            df_1h['Stoch_%K'], df_1h['Stoch_%D'] = calc_stoch(df_1h)
            stoch_1h_trend = "Stijgend 🟢" if df_1h['Stoch_%K'].iloc[-1] > df_1h['Stoch_%K'].iloc[-2] else "Dalend 🔴"
            stoch_1d_trend = "Stijgend 🟢" if stoch_k > stoch_k_prev else "Dalend 🔴"

            # 5. MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            macd_val = df['MACD'].iloc[-1]
            macd_prev = df['MACD'].iloc[-2]
            
            # 6. 3-Day Candlestick Analysis
            c3, c2, c1 = df['Close'].iloc[-3], df['Close'].iloc[-2], df['Close'].iloc[-1]
            o3, o2, o1 = df['Open'].iloc[-3], df['Open'].iloc[-2], df['Open'].iloc[-1]
            
            bullish_days = sum([1 for o, c in [(o3,c3), (o2,c2), (o1,c1)] if c > o])
            total_return_3d = ((c1 - o3) / o3) * 100
            
            if bullish_days == 3 and total_return_3d > 2:
                candle_signal = f"Zeer Sterk Bullish 🚀 (+{total_return_3d:.2f}%)"
            elif bullish_days >= 2 and total_return_3d > 0:
                candle_signal = f"Matig Bullish 📈 (+{total_return_3d:.2f}%)"
            elif total_return_3d < 0:
                candle_signal = f"Bearish 📉 ({total_return_3d:.2f}%)"
            else:
                candle_signal = f"Neutraal ➖ ({total_return_3d:.2f}%)"

            # 7. AI Model Simulation Scores (Proprietary Scoring Algoritme)
            # Momentum Score
            mom_score = 5.0
            if macd_val > 0: mom_score += 1.5
            if stoch_k > stoch_d: mom_score += 1.5
            if rsi_val > 55: mom_score += 1.0
            if vol_ratio > 1.2: mom_score += 1.0
            mom_score = min(10.0, max(1.0, mom_score))

            # Ensemble Score
            ensemble_score = round((mom_score * 0.6) + (10 - (stoch_d * 0.05)) * 0.4, 1)

            # ==================== DISPLAY DASHBOARD ====================
            
            st.subheader(f"Analyse Overzicht: {ticker_symbol}")
            
            # Top Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Huidige Koers", f"${c1:.2f}")
            col2.metric("Volume Ratio (vs 20D MA)", f"{vol_ratio:.2f}x", 
                        delta="Hoger dan normaal" if vol_ratio > 1.0 else "Lager dan normaal")
            col3.metric("AI Momentum Score", f"{mom_score:.1f} / 10")
            col4.metric("AI Ensemble Score", f"{ensemble_score:.1f} / 10")

            st.divider()

            # Technical Cards
            c_left, c_right = st.columns(2)

            with c_left:
                st.write("### 📊 Technische Indicatoren")
                
                # RSI Styling logic
                rsi_bg = "white"
                rsi_color = "black"
                if rsi_val > 70 and rsi_val < rsi_prev:
                    rsi_bg = "#ffcccc" # Rood (Overbought en daalt)
                    rsi_color = "#990000"
                elif rsi_val > 55:
                    rsi_bg = "#d4edda" # Groen
                    rsi_color = "#155724"
                
                st.markdown(f"""
                <div style="background-color:{rsi_bg}; color:{rsi_color}; padding:10px; border-radius:5px; margin-bottom:10px;">
                    <strong>RSI (14):</strong> {rsi_val:.2f}
                </div>
                """, unsafe_allow_html=True)

                # Stochastic Styling
                stoch_bg = "#d4edda" if (stoch_k > stoch_d and stoch_k > stoch_k_prev) else "#f8d7da"
                st.markdown(f"""
                <div style="background-color:{stoch_bg}; padding:10px; border-radius:5px; margin-bottom:10px;">
                    <strong>Stochastic (%K / %D):</strong> {stoch_k:.1f} / {stoch_d:.1f}
                </div>
                """, unsafe_allow_html=True)

                # MACD Logic
                if macd_val > 0 and macd_val > macd_prev:
                    macd_status = "🟢 Strong Buy (Boven 0 & Stijgend)"
                elif macd_val > 0:
                    macd_status = "🟢 Bullish (Boven 0)"
                else:
                    macd_status = "🔴 Bearish (Onder 0)"
                
                st.info(f"**MACD Status:** {macd_status}")

            with c_right:
                st.write("### 🔍 Sentiment & Trends")
                
                # Put / Call Ratio
                if put_call_ratio:
                    pc_status = "Bullish (< 0.8) 🟢" if put_call_ratio < 0.8 else ("Bearish (> 1.0) 🔴" if put_call_ratio > 1.0 else "Neutraal 🟡")
                    st.write(f"**Put/Call Ratio:** {put_call_ratio:.2f} → *{pc_status}*")
                else:
                    st.write("**Put/Call Ratio:** Niet beschikbaar voor deze ticker")

                # Short Float
                if short_float:
                    st.write(f"**Short Float:** {short_float * 100:.2f}%")
                else:
                    st.write("**Short Float:** Niet beschikbaar")

                st.write(f"**Stochastic Trend 1 Uur:** {stoch_1h_trend}")
                st.write(f"**Stochastic Trend 1 Dag:** {stoch_1d_trend}")
                st.write(f"**3-Daagse Candle Analyse:** {candle_signal}")

    except Exception as e:
        st.error(f"Fout bij het ophalen van gegevens: {e}")
