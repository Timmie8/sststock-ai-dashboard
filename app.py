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

def get_signal_and_color(score):
    """
    Bepaalt het advies en de kleur/emoji op basis van de Totaal Score:
    - 7.0 t/m 10.0 : BUY / LONG (Groen)
    - 5.0 t/m 6.9  : WATCH (Oranje)
    - Onder 5.0    : AVOID (Rood)
    """
    if score >= 7.0:
        return "BUY / LONG 🟢", "#d4edda", "#155724" # Groen
    elif score >= 5.0:
        return "WATCH 🟠", "#fff3cd", "#856404"     # Oranje
    else:
        return "AVOID 🔴", "#f8d7da", "#721c24"     # Rood

def calculate_composite_score(vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float, mfi_val, ad_trend_3d):
    """
    Berekent de Totaal Score (1-10) op basis van gewogen pijlers inclusief MFI en A/D:
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

        # Put/Call Ratio Bepaling
        put_call_ratio = info.get('putCallRatio', None)
        pcr_source = "Info"

        if put_call_ratio is None or np.isnan(put_call_ratio):
            try:
                options_dates = stock.options
                if options_dates:
                    near_option = stock.option_chain(options_dates[0])
                    calls = near_option.calls
                    puts = near_option.puts
                    
                    total_call_vol = calls['volume'].sum()
                    total_put_vol = puts['volume'].sum()

                    if total_call_vol > 0:
                        put_call_ratio = total_put_vol / total_call_vol
                        pcr_source = "Volume (Chain)"
                    else:
                        total_call_oi = calls['openInterest'].sum()
                        total_put_oi = puts['openInterest'].sum()
                        if total_call_oi > 0:
                            put_call_ratio = total_put_oi / total_call_oi
                            pcr_source = "OI (Chain)"
            except Exception:
                put_call_ratio = None

        short_float = info.get('shortPercentOfFloat', None)

        # 1. Volume Ratio
        avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        vol_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # 2. Money Flow Index (MFI - 1 Day / 14-period indicator value)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        
        pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
        neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
        
        pos_mf14 = pos_flow.rolling(14).sum()
        neg_mf14 = neg_flow.rolling(14).sum()
        
        mfi = 100 - (100 / (1 + (pos_mf14 / neg_mf14)))
        mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50.0

        # MFI Score & Kleur
        if mfi_val >= 60:
            mfi_score, mfi_bg, mfi_color = round(min(10.0, 5.0 + (mfi_val - 50) / 5), 1), "#d4edda", "#155724" # Instroom / Bullish
        elif mfi_val <= 40:
            mfi_score, mfi_bg, mfi_color = round(max(1.0, 5.0 - (50 - mfi_val) / 5), 1), "#f8d7da", "#721c24"  # Uitstroom / Bearish
        else:
            mfi_score, mfi_bg, mfi_color = 5.0, "#fff3cd", "#856404"                                           # Neutraal

        # 3. 3-Daagse Accumulatie / Distributie (A/D)
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv = clv.fillna(0)
        ad_line = (clv * df['Volume']).cumsum()
        
        ad_diff_3d = ad_line.iloc[-1] - ad_line.iloc[-4]
        if ad_diff_3d > 0:
            ad_trend_3d = "Accumulatie 🟢"
            ad_score, ad_bg, ad_color = 8.5, "#d4edda", "#155724"
        elif ad_diff_3d < 0:
            ad_trend_3d = "Distributie 🔴"
            ad_score, ad_bg, ad_color = 3.0, "#f8d7da", "#721c24"
        else:
            ad_trend_3d = "Neutraal 🟡"
            ad_score, ad_bg, ad_color = 5.0, "#fff3cd", "#856404"

        # 4. Support & Resistance Bepaling (Pivot Points & Swing High/Low)
        high_20 = df['High'].iloc[-21:-1].max()
        low_20 = df['Low'].iloc[-21:-1].min()
        last_close = df['Close'].iloc[-1]

        # Pivots op basis van laatste gesloten dag
        prev_high = df['High'].iloc[-2]
        prev_low = df['Low'].iloc[-2]
        prev_close = df['Close'].iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3
        
        resistance_1 = (2 * pivot) - prev_low
        support_1 = (2 * pivot) - prev_high

        # Gebruik de meest relevante niveaus
        effective_resistance = min(high_20, resistance_1) if min(high_20, resistance_1) > last_close else max(high_20, resistance_1)
        effective_support = max(low_20, support_1) if max(low_20, support_1) < last_close else min(low_20, support_1)

        # 5. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        rsi_val = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-2]

        # 6. Stochastic Oscillator
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

        # 7. MACD
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

        # 8. 3-Day Candlestick Analysis
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

        # 9. AI Model Simulation
        mom_score = 5.0
        if macd_val > 0: mom_score += 1.5
        if stoch_k > stoch_d: mom_score += 1.5
        if rsi_val > 55: mom_score += 1.0
        if vol_ratio > 1.2: mom_score += 1.0
        mom_score = min(10.0, max(1.0, mom_score))

        ensemble_score = round((mom_score * 0.6) + (10 - (stoch_d * 0.05)) * 0.4, 1)

        # 10. Totaal Score Berekening
        totaal_score, vol_score, pcr_score, sent_score, tech_score = calculate_composite_score(
            vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float, mfi_val, ad_trend_3d
        )

        signal, bg_color, text_color = get_signal_and_color(totaal_score)

        return {
            "Ticker": symbol,
            "Koers": f"${c1:.2f}",
            "Totaal Score": totaal_score,
            "Advies": signal,
            "AI Ensemble": ensemble_score,
            "AI Momentum": round(mom_score, 1),
            "Volume Ratio": f"{vol_ratio:.2f}x",
            "1D MoneyFlow": mfi_val,
            "MFI Score": mfi_score,
            "MFI BG": mfi_bg,
            "MFI Color": mfi_color,
            "3D Acc/Dist": ad_trend_3d,
            "3D AD Score": ad_score,
            "3D AD BG": ad_bg,
            "3D AD Color": ad_color,
            "Support": f"${effective_support:.2f}",
            "Resistance": f"${effective_resistance:.2f}",
            "RSI (14)": round(rsi_val, 1),
            "RSI Raw": rsi_val,
            "RSI Prev": rsi_prev,
            "MACD Status": macd_status,
            "Put/Call": f"{put_call_ratio:.2f}" if (put_call_ratio is not None and not np.isnan(put_call_ratio)) else "Geen Opties",
            "PCR Raw": put_call_ratio,
            "PCR Bron": pcr_source,
            "Short Float": f"{short_float * 100:.1f}%" if (short_float is not None and not np.isnan(short_float)) else "N/B",
            "Stoch %K": round(stoch_k, 1),
            "Stoch %D": round(stoch_d, 1),
            "Stoch K Prev": stoch_k_prev,
            "Stoch 1H": stoch_1h_trend,
            "Stoch 1D": stoch_1d_trend,
            "3D Candles": candle_signal,
            "Tech Score": tech_score,
            "Volume Score": vol_score,
            "PCR Score": pcr_score,
            "Sentiment Score": sent_score,
            "Advies BG": bg_color,
            "Advies Text": text_color
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

        # Sorteren op Totaal Score (hoogste bovenaan)
        df_results = df_results.sort_values(by="Totaal Score", ascending=False)

        st.subheader("📊 Multi-Stock Scan Overzicht")
        
        # Weergave tabel voorbereiden
        display_cols = [
            "Ticker", "Koers", "Totaal Score", "Advies", "1D MoneyFlow", "3D Acc/Dist", 
            "Support", "Resistance", "Volume Ratio", "RSI (14)", "MACD Status", 
            "Put/Call", "Short Float"
        ]

        st.dataframe(
            df_results[display_cols],
            column_config={
                "Totaal Score": st.column_config.NumberColumn(format="%.1f 🏆"),
                "1D MoneyFlow": st.column_config.NumberColumn(format="%.1f MFI"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        # Inzoomen op 1 specifiek aandeel
        selected_ticker = st.selectbox("Selecteer een aandeel voor gedetailleerde kaartweergave:", df_results["Ticker"])
        selected_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

        if selected_data:
            st.write(f"### 🔍 Gedetailleerde Analyse voor {selected_ticker}")
            
            # Gekleurde Advies Banner
            st.markdown(f"""
            <div style="background-color:{selected_data['Advies BG']}; color:{selected_data['Advies Text']}; padding:15px; border-radius:8px; font-size:20px; font-weight:bold; text-align:center; margin-bottom:15px;">
                Totaal Score: {selected_data['Totaal Score']} / 10 — Advies: {selected_data['Advies']}
            </div>
            """, unsafe_allow_html=True)

            # Top Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Koers", selected_data["Koers"])
            m2.metric("Support (Steun)", selected_data["Support"])
            m3.metric("Resistance (Weerstand)", selected_data["Resistance"])
            m4.metric("Volume Ratio", selected_data["Volume Ratio"])
            m5.metric("RSI (14)", selected_data["RSI (14)"])

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                st.write("#### 📊 Money Flow & Accumulatie")

                # 1D Money Flow Custom Box
                st.markdown(f"""
                <div style="background-color:{selected_data['MFI BG']}; color:{selected_data['MFI Color']}; padding:12px; border-radius:6px; margin-bottom:10px;">
                    <strong>1D Money Flow Index (MFI):</strong> {selected_data['1D MoneyFlow']} 
                    <br><em>Score: {selected_data['MFI Score']} / 10</em>
                </div>
                """, unsafe_allow_html=True)

                # 3D Acc/Dist Custom Box
                st.markdown(f"""
                <div style="background-color:{selected_data['3D AD BG']}; color:{selected_data['3D AD Color']}; padding:12px; border-radius:6px; margin-bottom:10px;">
                    <strong>3-Daagse Acc / Distributie:</strong> {selected_data['3D Acc/Dist']} 
                    <br><em>Score: {selected_data['3D AD Score']} / 10</em>
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

            with col_right:
                st.write("#### 🔍 Sentiment & Key Levels")

                st.info(f"**MACD Status:** {selected_data['MACD Status']}")

                pcr_val = selected_data["PCR Raw"]
                if pcr_val is not None and not np.isnan(pcr_val):
                    pc_status = "Bullish (< 0.8) 🟢" if pcr_val < 0.8 else ("Bearish (> 1.0) 🔴" if pcr_val > 1.0 else "Neutraal 🟡")
                    st.write(f"**Put/Call Ratio:** {pcr_val:.2f} (*{selected_data['PCR Bron']}*) → *{pc_status}*")
                else:
                    st.write("**Put/Call Ratio:** Geen optie-data beschikbaar")

                st.write(f"**Short Float:** {selected_data['Short Float']}")
                st.write(f"**Stochastic Trend 1 Uur:** {selected_data['Stoch 1H']}")
                st.write(f"**Stochastic Trend 1 Dag:** {selected_data['Stoch 1D']}")
                st.write(f"**3-Daagse Candle Analyse:** {selected_data['3D Candles']}")

            st.write("---")
            st.write("#### 🎯 Score Opbouw Breakdown")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Technische & Acc Score (40%)", f"{selected_data['Tech Score']} / 10")
            sc2.metric("Volume & MoneyFlow Score (25%)", f"{selected_data['Volume Score']} / 10")
            sc3.metric("Sentiment Score (20%)", f"{selected_data['Sentiment Score']} / 10")
            sc4.metric("Opties/PCR Score (15%)", f"{selected_data['PCR Score']} / 10")
