import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')

# Streamlit Pagina Configuratie
st.set_page_config(
    page_title="AI Swingtrade & ML Probability Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live AI & ML Swingtrade Scanner (1–5 Dagen)")
st.caption("Geavanceerde analyse met ML-stijgingskans (3 dagen), Volume Breakouts, Short Float, Money Flow en Technische Indicatoren.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Instellingen & Watchlist")
user_input = st.sidebar.text_input(
    "Vul tickers in (gescheiden door komma's):",
    value="NVDA, TSLA, AMD, PLTR, AAPL"
)

tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
scan_button = st.sidebar.button("🚀 Start Live Scan", type="primary")


# --- ML PROBABILITY & COMPOSITE SCORE ENGINE ---
def calculate_ml_3d_probability(rsi_val, macd_diff, vol_ratio, mfi_val, ad_trend_3d, ema5, ema15, live_price):
    base_prob = 50.0  # Neutrale startkans

    # Feature 1: Trend Alignment (EMA 5 vs 15 vs Price)
    if live_price > ema5 > ema15:
        base_prob += 10.0
    elif live_price < ema5 < ema15:
        base_prob -= 10.0

    # Feature 2: Momentum & MACD
    if macd_diff > 0:
        base_prob += 8.0
    else:
        base_prob -= 6.0

    # Feature 3: Volume & Money Flow (MFI)
    if vol_ratio >= 1.5 and mfi_val >= 55:
        base_prob += 12.0
    elif vol_ratio >= 1.2:
        base_prob += 5.0
    elif vol_ratio < 0.8:
        base_prob -= 5.0

    # Feature 4: Accumulatie / Distributie Trend
    if ad_trend_3d == "Accumulatie 🟢":
        base_prob += 8.0
    elif ad_trend_3d == "Distributie 🔴":
        base_prob -= 8.0

    # Feature 5: RSI Sweet Spot (45 - 65)
    if 48 <= rsi_val <= 62:
        base_prob += 7.0
    elif rsi_val > 70:
        base_prob -= 8.0  
    elif rsi_val < 35:
        base_prob += 3.0  

    final_prob = min(95.0, max(15.0, base_prob))
    return round(final_prob, 1)


def calculate_comprehensive_scores(vol_ratio, rsi_val, macd_val, macd_prev, stoch_k, stoch_d, put_call_ratio, short_float, mfi_val, ad_trend_3d, ml_prob):
    # 1. Volume & Money Flow Score (25%)
    vol_score = 5.0
    if vol_ratio >= 1.5: vol_score += 2.5
    elif vol_ratio >= 1.1: vol_score += 1.0

    if mfi_val >= 60: vol_score += 2.5
    elif mfi_val >= 45: vol_score += 1.0
    elif mfi_val < 35: vol_score -= 1.5
    vol_score = round(min(10.0, max(1.0, vol_score)), 1)

    # 2. Opties / PCR Score (15%)
    pcr_score = 5.0
    if put_call_ratio is not None and not np.isnan(put_call_ratio):
        if put_call_ratio < 0.8: pcr_score = 9.0
        elif put_call_ratio <= 1.0: pcr_score = 6.5
        else: pcr_score = 3.0
    pcr_score = round(pcr_score, 1)

    # 3. Sentiment & Short Interest Score (15%)
    sentiment_score = 5.0
    if short_float is not None and not np.isnan(short_float):
        if short_float < 0.05: sentiment_score += 2.5
        elif short_float > 0.15: sentiment_score -= 2.0
    if vol_ratio > 1.3: sentiment_score += 1.5
    sentiment_score = round(min(10.0, max(1.0, sentiment_score)), 1)

    # 4. Technische Indicatoren & Accumulatie Score (25%)
    tech_score = 5.0
    if 55 <= rsi_val <= 70: tech_score += 1.0
    elif rsi_val > 70: tech_score -= 1.0
    if macd_val > 0 and macd_val > macd_prev: tech_score += 1.5
    if stoch_k > stoch_d: tech_score += 1.0
    if ad_trend_3d == "Accumulatie 🟢": tech_score += 1.5
    elif ad_trend_3d == "Distributie 🔴": tech_score -= 1.5
    tech_score = round(min(10.0, max(1.0, tech_score)), 1)

    # 5. ML Probability Impact Score (20%)
    ml_score = round(ml_prob / 10.0, 1)

    # Totaal Gewogen AI Score (1-10)
    total_score = (
        (vol_score * 0.25) +
        (pcr_score * 0.15) +
        (sentiment_score * 0.15) +
        (tech_score * 0.25) +
        (ml_score * 0.20)
    )
    total_score = round(total_score, 1)

    return total_score, vol_score, pcr_score, sentiment_score, tech_score, ml_score


# --- ANALYSE FUNCTIE ---
def get_live_swing_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval="1d")
        if df.empty or len(df) < 30:
            return None
            
        info = ticker.info
        live_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        day_change_pct = ((live_price - prev_close) / prev_close) * 100
        
        # Volume Spike & Ratio
        current_volume = df['Volume'].iloc[-1]
        avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = current_volume / avg_vol_20d if avg_vol_20d > 0 else 1.0
        
        # 1D Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
        neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
        pos_mf14 = pos_flow.rolling(14).sum()
        neg_mf14 = neg_flow.rolling(14).sum()
        mfi = 100 - (100 / (1 + (pos_mf14 / neg_mf14)))
        mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50.0

        mfi_status = "Bullish 🟢" if mfi_val >= 60 else ("Bearish 🔴" if mfi_val <= 40 else "Neutraal 🟡")

        # 3D Accumulatie / Distributie
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv = clv.fillna(0)
        ad_line = (clv * df['Volume']).cumsum()
        ad_diff_3d = ad_line.iloc[-1] - ad_line.iloc[-4]
        
        ad_trend_3d = "Accumulatie 🟢" if ad_diff_3d > 0 else ("Distributie 🔴" if ad_diff_3d < 0 else "Neutraal 🟡")

        # Support & Resistance
        high_20 = df['High'].iloc[-21:-1].max()
        low_20 = df['Low'].iloc[-21:-1].min()
        prev_high = df['High'].iloc[-2]
        prev_low = df['Low'].iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3
        resistance_1 = (2 * pivot) - prev_low
        support_1 = (2 * pivot) - prev_high

        effective_resistance = min(high_20, resistance_1) if min(high_20, resistance_1) > live_price else max(high_20, resistance_1)
        effective_support = max(low_20, support_1) if max(low_20, support_1) < live_price else min(low_20, support_1)

        # Technische Indicatoren
        ema5 = ta.trend.ema_indicator(df['Close'], window=5).iloc[-1]
        ema15 = ta.trend.ema_indicator(df['Close'], window=15).iloc[-1]
        rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
        
        low_min14 = df['Low'].rolling(window=14).min()
        high_max14 = df['High'].rolling(window=14).max()
        stoch_k_series = 100 * ((df['Close'] - low_min14) / (high_max14 - low_min14))
        stoch_d_series = stoch_k_series.rolling(window=3).mean()
        stoch_k = stoch_k_series.iloc[-1]
        stoch_d = stoch_d_series.iloc[-1]

        macd = ta.trend.MACD(df['Close'])
        macd_val = macd.macd().iloc[-1]
        macd_prev = macd.macd().iloc[-2]
        macd_diff = macd.macd_diff().iloc[-1]

        # Put/Call Ratio
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

        short_pct = info.get('shortPercentOfFloat', 0) or 0
        sector = info.get('sector', 'Onbekend')

        # Bereken ML Stijgingskans (3 dagen)
        ml_prob_3d = calculate_ml_3d_probability(
            rsi_val=rsi,
            macd_diff=macd_diff,
            vol_ratio=vol_ratio,
            mfi_val=mfi_val,
            ad_trend_3d=ad_trend_3d,
            ema5=ema5,
            ema15=ema15,
            live_price=live_price
        )

        # Bereken Gewogen Scores
        total_score, vol_score, pcr_score, sent_score, tech_score, ml_score = calculate_comprehensive_scores(
            vol_ratio=vol_ratio, rsi_val=rsi, macd_val=macd_val, macd_prev=macd_prev,
            stoch_k=stoch_k, stoch_d=stoch_d, put_call_ratio=pcr_volume, short_float=short_pct,
            mfi_val=mfi_val, ad_trend_3d=ad_trend_3d, ml_prob=ml_prob_3d
        )

        # 3D Candlestick Signal
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
            "ML Kans Stijging (3d)": f"{ml_prob_3d}%",
            "ML Prob Raw": ml_prob_3d,
            "Totaal Score": total_score,
            "Signaal": "BUY / LONG 🟢" if total_score >= 7.0 else ("WATCH 🟠" if total_score >= 5.0 else "AVOID / SHORT 🔴"),
            "Short Float Raw": short_pct,
            "1D MoneyFlow": mfi_status,
            "3D Acc/Dist": ad_trend_3d,
            "3D Candles": candle_signal,
            "Support": f"${effective_support:.2f}",
            "Resistance": f"${effective_resistance:.2f}",
            "Volume Ratio": f"{round(vol_ratio, 2)}x",
            "RSI Raw": round(rsi, 1),
            "EMA Trend": "Bullish" if ema5 > ema15 else "Bearish",
            "Put/Call Ratio": f"{round(pcr_volume, 2)} ({pcr_status})" if pcr_volume else "N/B",
            "Tech Score": tech_score,
            "Volume Score": vol_score,
            "PCR Score": pcr_score,
            "Sentiment Score": sent_score,
            "ML Score": ml_score,
            "Entry": f"${entry}",
            "Stop Loss (-3.5%)": f"${sl}",
            "TP1 (1-3d)": f"${tp1}",
            "TP2 (3-5d)": f"${tp2}"
        }
    except Exception as e:
        return None


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
        
        # Sorteren op de hoogste ML Stijgingskans (%)
        df_res = df_res.sort_values(by="ML Prob Raw", ascending=False)
        
        st.subheader("📊 Ranking: Hoogste ML Kans op Stijging (3 Dagen)")

        # Kolommen voor weergave insluiten
        display_df = df_res[[
            "Ticker", "ML Kans Stijging (3d)", "Totaal Score", "Signaal", "Koers", "Verandering", 
            "RSI Raw", "Short Float Raw", "1D MoneyFlow", "3D Acc/Dist", "3D Candles", 
            "Support", "Resistance", "Volume Ratio", "Put/Call Ratio"
        ]].copy()

        display_df.rename(columns={
            "RSI Raw": "RSI",
            "Short Float Raw": "Short Float"
        }, inplace=True)

        # STYLING FUNCTIES FOR VAKJES KLEUREN
        def highlight_rsi(val):
            """Kleurt RSI vakje groen als boven 55"""
            if pd.notnull(val) and val > 55:
                return 'background-color: #2e7d32; color: white; font-weight: bold;'
            return ''

        def highlight_short_float(val):
            """Kleurt Short Float vakje rood als boven 10% (0.10)"""
            if pd.notnull(val) and val > 0.10:
                return 'background-color: #c62828; color: white; font-weight: bold;'
            return ''

        # Pas de Styler toe op het DataFrame
        styled_df = display_df.style.applymap(highlight_rsi, subset=['RSI']) \
                                   .applymap(highlight_short_float, subset=['Short Float']) \
                                   .format({
                                       'Short Float': '{:.1%}',
                                       'RSI': '{:.1f}'
                                   })

        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True
        )
        
        st.subheader("🎯 Concrete Trade Setups & Score Breakdown")
        for item in df_res.to_dict('records'):
            with st.expander(f"{item['Ticker']} — ML Stijgingskans: {item['ML Kans Stijging (3d)']} | Totaal Score: {item['Totaal Score']}/10 ({item['Signaal']})"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ML Kans op Stijging (3d)", item["ML Kans Stijging (3d)"])
                c2.metric("Entry Level", item["Entry"])
                c3.metric("Stop Loss (-3.5%)", item["Stop Loss (-3.5%)"])
                c4.metric("Take Profit 1 (1-3d)", item["TP1 (1-3d)"])
                
                st.write("---")
                st.write("#### 🔍 Transparante Score Opbouw (1-10 per categorie)")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("ML Model (20%)", f"{item['ML Score']} / 10")
                sc2.metric("Tech & Acc (25%)", f"{item['Tech Score']} / 10")
                sc3.metric("Volume & Flow (25%)", f"{item['Volume Score']} / 10")
                sc4.metric("Opties/PCR (15%)", f"{item['PCR Score']} / 10")
                sc5.metric("Sentiment (15%)", f"{item['Sentiment Score']} / 10")

                st.write("---")
                st.write(f"**Support:** {item['Support']} | **Resistance:** {item['Resistance']}")
                st.write(f"**Sector:** {item['Sector']} | **EMA Trend:** {item['EMA Trend']} | **Volume Ratio:** {item['Volume Ratio']} | **Short Float:** {item['Short Float Raw']:.1%}")
