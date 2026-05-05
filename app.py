import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from engine import SentimentEngine
from datetime import datetime, timedelta

st.set_page_config(page_title="News Trends Dashboard", layout="wide", initial_sidebar_state="expanded")

def local_css():
    st.markdown("""
        <style>
        .stApp { background: radial-gradient(circle at 20% 30%, #1c2a3a 0%, #0e1117 100%); }
        .glass-card {
            background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px;
            padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .filter-card {
            background: rgba(0, 255, 163, 0.05); border: 1px solid rgba(0, 255, 163, 0.2);
            border-radius: 10px; padding: 15px; margin-bottom: 20px;
        }
        h1, h2, h3 { color: #00ffa3 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
        .metric-label { color: #888; font-size: 0.9rem; text-transform: uppercase; }
        .metric-value { color: #ffffff; font-size: 1.8rem; font-weight: 800; }
        [data-testid="stSidebar"] { background-color: rgba(14, 17, 23, 0.95); border-right: 1px solid rgba(255, 255, 255, 0.1); }
        .stButton>button { background: linear-gradient(90deg, #00ffa3 0%, #00d1ff 100%); color: #0e1117 !important; border: none; font-weight: bold; border-radius: 8px; transition: all 0.3s ease; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0, 255, 163, 0.4); }
        .feed-card { background: rgba(255, 255, 255, 0.03); border-left: 4px solid #00ffa3; padding: 15px; margin-bottom: 10px; border-radius: 0 10px 10px 0; }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_engine(api_key=None):
    return SentimentEngine(api_key=api_key)

@st.cache_data(ttl=3600)
def fetch_and_analyze(_engine, keyword, limit, days_back):
    df = _engine.fetch_market_data(keyword, limit, days_back)
    df = _engine.analyze_intelligence(df, keyword)
    stats = _engine.aggregate_market_insights(df)

    # Ensure dates are properly parsed for filtering later
    if not df.empty:
        df['date'] = pd.to_datetime(df['publishedAt']).dt.date
    return df, stats

def init_session_state():
    """Initialize state variables for cross-filtering"""
    if 'active_aspect' not in st.session_state: st.session_state.active_aspect = None
    if 'active_sources' not in st.session_state: st.session_state.active_sources = []
    if 'active_sentiments' not in st.session_state: st.session_state.active_sentiments = []
    if 'active_dates' not in st.session_state: st.session_state.active_dates = ()
    if 'applied_days' not in st.session_state: st.session_state.applied_days = 7
    if 'applied_limit' not in st.session_state: st.session_state.applied_limit = 50

def reset_filters():
    """Clear all interactive filters"""
    st.session_state.active_aspect = None
    st.session_state.active_sources = []
    st.session_state.active_sentiments = []
    st.session_state.active_dates = ()

def main():
    local_css()
    init_session_state()

    with st.sidebar:
        st.title("Intelligence")
        st.markdown("News Research & Sentiment Engine")
        st.markdown("---")

        keyword = st.text_input("Analysis Target", "Artificial Intelligence")

        col1, col2 = st.columns(2)
        with col1:
            days_back = st.number_input("Days Back", min_value=1, max_value=30, value=7)
        with col2:
            limit = st.number_input("Data Depth", min_value=10, max_value=100, value=50)

        news_api_key = st.text_input("NewsAPI Key", type="password", help="Leave blank for simulation")

        st.markdown("---")
        analyze_btn = st.button("Generate Report", type="primary", use_container_width=True)

        if not news_api_key:
            st.caption("⚠️ No API Key provided. Running in High-Fidelity Simulation Mode.")

    st.title("News Trends Dashboard")
    st.markdown(f"Real-time sentiment and aspect tracking for: **{keyword}**")

    if analyze_btn:
        st.session_state.target = keyword
        st.session_state.applied_days = days_back
        st.session_state.applied_limit = limit
        reset_filters()

    if 'target' in st.session_state:
        engine = get_engine(news_api_key)

        with st.spinner("Synchronizing News data and performing NLP analysis..."):
            # Use applied parameters to avoid re-triggering analysis on every widget click
            df, stats = fetch_and_analyze(engine, st.session_state.target, st.session_state.applied_limit, st.session_state.applied_days)

        if df.empty:
            st.warning("No data found for this keyword in the given timeframe.")
            return

        # --- TOP METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        pos_pct = (df['sentiment'] == 'POSITIVE').mean() * 100
        avg_conf = df['confidence'].mean() * 100

        with m1: st.markdown(f'<div class="glass-card"><p class="metric-label">Volume</p><p class="metric-value">{len(df)}</p></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="glass-card"><p class="metric-label">Sentiment Pulse</p><p class="metric-value" style="color:{"#00ffa3" if pos_pct >= 50 else "#ff4b4b"}">{pos_pct:.1f}%</p></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="glass-card"><p class="metric-label">Model Confidence</p><p class="metric-value">{avg_conf:.1f}%</p></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="glass-card"><p class="metric-label">News Alpha</p><p class="metric-value" style="color:#00d1ff">{"Bullish" if pos_pct > 60 else "Bearish" if pos_pct < 40 else "Neutral"}</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 1: GAUGE & TOP DRIVERS ---
        col_gauge, col_aspect = st.columns([1, 2])

        with col_gauge:
            st.subheader("News Sentiment")
            fig_gauge = go.Figure(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[(df['sentiment']=='POSITIVE').sum(), (df['sentiment']=='NEGATIVE').sum(), (df['sentiment']=='NEUTRAL').sum()],
                hole=.75, marker_colors=['#00ffa3', '#ff4b4b', '#888888']
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=250, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            fig_gauge.add_annotation(text=f"{pos_pct:.0f}%", x=0.5, y=0.5, font_size=32, showarrow=False, font_color="#00ffa3" if pos_pct >= 50 else "#ff4b4b")
            st.plotly_chart(fig_gauge, width='stretch')

        with col_aspect:
            st.subheader("Top Drivers (Aspects)")
            if not stats.empty:
                top_n = stats.sort_values('total', ascending=False).head(8).reset_index()
                fig_div = go.Figure()
                fig_div.add_trace(go.Bar(y=top_n['aspect'], x=top_n['POSITIVE'], name='Positive', orientation='h', marker_color='#00ffa3'))
                fig_div.add_trace(go.Bar(y=top_n['aspect'], x=-top_n['NEGATIVE'], name='Negative', orientation='h', marker_color='#ff4b4b'))
                fig_div.update_layout(barmode='relative', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=250, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='white'), yaxis=dict(autorange="reversed"))
                
                sel_div = st.plotly_chart(fig_div, width='stretch', on_select="rerun", selection_mode="points")
                if sel_div and sel_div.selection.points:
                    clicked_aspect = sel_div.selection.points[0]["y"]
                    if st.session_state.active_aspect != clicked_aspect:
                        st.session_state.active_aspect = clicked_aspect
                        st.toast(f"Filtering by Topic: {clicked_aspect}")

        # --- ROW 2: IMPROVED PRIORITY MATRIX ---
        st.markdown("---")
        st.subheader("Priority Matrix: Impact vs. Sentiment")
        if not stats.empty:
            stats_reset = stats.reset_index().groupby(['total', 'sentiment_score']).agg({
                'aspect': lambda x: ' & '.join(x),
                'category': lambda x: ' / '.join(x.unique())
            }).reset_index()

            volume_threshold = stats_reset['total'].quantile(0.80)
            stats_reset['display_label'] = stats_reset.apply(
                lambda x: x['aspect'] if x['total'] >= volume_threshold else "", axis=1
            )

            fig_scatter = px.scatter(
                stats_reset, x='total', y='sentiment_score', size='total', color='sentiment_score',
                hover_name='aspect', text='display_label',
                color_continuous_scale=['#ff4b4b', '#333333', '#00ffa3'],
                labels={'total': 'Mentions (Impact)', 'sentiment_score': 'Sentiment Index'},
                template="plotly_dark", size_max=45
            )

            fig_scatter.update_traces(
                textposition='top center',
                textfont=dict(size=12, color="#ffffff", family="Arial Black"),
                marker=dict(line=dict(width=1.5, color='rgba(255,255,255,0.6)'))
            )

            median_vol = stats_reset['total'].median() if not stats_reset.empty else 1
            max_vol = stats_reset['total'].max() if not stats_reset.empty else 10
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
            fig_scatter.add_vline(x=median_vol, line_dash="dash", line_color="white", opacity=0.3)

            # Add Quadrant Labels
            fig_scatter.add_annotation(x=max_vol, y=1.3, text="STRATEGIC WINS", showarrow=False, font=dict(color="#00ffa3", size=11, family="Arial Black"), xanchor="right")
            fig_scatter.add_annotation(x=max_vol, y=-1.1, text="CRITICAL RISKS", showarrow=False, font=dict(color="#ff4b4b", size=11, family="Arial Black"), xanchor="right")
            fig_scatter.add_annotation(x=0, y=1.3, text="EMERGING TRENDS", showarrow=False, font=dict(color="#00d1ff", size=11, family="Arial Black"), xanchor="left")
            fig_scatter.add_annotation(x=0, y=-1.1, text="LOW PRIORITY", showarrow=False, font=dict(color="#888888", size=11, family="Arial Black"), xanchor="left")

            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=0, l=0, r=0), height=500,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[-1.3, 1.5]),
                coloraxis_showscale=False
            )

            # Interaction: Update Aspect Filter
            sel_scatter = st.plotly_chart(fig_scatter, width='stretch', on_select="rerun", selection_mode="points")
            if sel_scatter and sel_scatter.selection.points:
                pt_idx = sel_scatter.selection.points[0]["point_index"]
                clicked_aspect = stats_reset.iloc[pt_idx]['aspect']
                if st.session_state.active_aspect != clicked_aspect:
                    st.session_state.active_aspect = clicked_aspect
                    st.toast(f"Topic selected: {clicked_aspect}")

        # --- ROW 3: TEMPORAL & SHARE OF VOICE (INTERACTIVE) ---
        st.markdown("---")
        st.markdown("### Deep analysis")
        st.caption("Click on any bar or pie slice below to automatically filter the news feed.")
        col_time, col_sov = st.columns([2, 1])

        with col_time:
            time_stats = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            fig_time = px.bar(
                time_stats, x='date', y='count', color='sentiment',
                color_discrete_map={'POSITIVE': '#00ffa3', 'NEGATIVE': '#ff4b4b', 'NEUTRAL': '#888888'},
                labels={'date': 'Date', 'count': 'Article Volume', 'sentiment': 'Sentiment'},
                template="plotly_dark", title="Temporal Sentiment"
            )

            fig_time.update_layout(
                barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(t=30, b=10, l=10, r=10),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Interaction: Update Date Filter
            sel_time = st.plotly_chart(fig_time, width='stretch', on_select="rerun", selection_mode="points")
            if sel_time and sel_time.selection.points:
                clicked_date_str = sel_time.selection.points[0]["x"]
                clicked_date = pd.to_datetime(clicked_date_str).date()
                if st.session_state.active_dates != (clicked_date, clicked_date):
                    st.session_state.active_dates = (clicked_date, clicked_date)
                    st.toast(f"Date range set to: {clicked_date}")

        with col_sov:
            # FIX: Pre-aggregate SOV Data so point_index works correctly
            sov_stats = df['source'].value_counts().reset_index()
            sov_stats.columns = ['source', 'count']

            fig_sov = px.pie(
                sov_stats, names='source', values='count', hole=0.5, template="plotly_dark", title="Share of Voice"
            )
            fig_sov.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0e1117', width=2)))
            fig_sov.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)

            # Interaction: Update Source Filter
            sel_sov = st.plotly_chart(fig_sov, width='stretch', on_select="rerun", selection_mode="points")
            if sel_sov and sel_sov.selection.points:
                # For Pie charts, the label is usually in 'label' or 'label'
                clicked_source = sel_sov.selection.points[0].get("label") or sov_stats.iloc[sel_sov.selection.points[0]["point_index"]]['source']

                if [clicked_source] != st.session_state.active_sources:
                    st.session_state.active_sources = [clicked_source]
                    st.toast(f"Filtered by Source: {clicked_source}")

        # --- ROW 4: INTELLIGENCE FEED WITH FILTERS ---
        st.markdown("---")

        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        st.subheader("Filter & Feed Console")

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])

        min_d, max_d = df['date'].min(), df['date'].max()
        all_sources = sorted(df['source'].unique().tolist())

        with col_f1:
            st.session_state.active_sentiments = st.multiselect("Sentiment", ["POSITIVE", "NEUTRAL", "NEGATIVE"], default=st.session_state.active_sentiments)
        with col_f2:
            st.session_state.active_sources = st.multiselect("News Source", all_sources, default=st.session_state.active_sources)
        with col_f3:
            default_dates = st.session_state.active_dates if st.session_state.active_dates else (min_d, max_d)
            st.session_state.active_dates = st.date_input("Date Range", value=default_dates, min_value=min_d, max_value=max_d)
        with col_f4:
            st.write("")
            st.write("")
            if st.button("❌ Clear Filters", type="secondary", use_container_width=True):
                reset_filters()
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Apply Filters sequentially
        final_df = df.copy()

        if st.session_state.active_aspect:
            st.success(f"🎯 **Drill-down Active:** Filtering for articles related to '{st.session_state.active_aspect}'")
            pattern = "|".join(st.session_state.active_aspect.split(" & "))
            final_df = final_df[final_df['text'].str.contains(pattern, case=False, na=False) | final_df['title'].str.contains(pattern, case=False, na=False)]

        if st.session_state.active_sentiments:
            final_df = final_df[final_df['sentiment'].isin(st.session_state.active_sentiments)]

        if st.session_state.active_sources:
            final_df = final_df[final_df['source'].isin(st.session_state.active_sources)]

        if st.session_state.active_dates and len(st.session_state.active_dates) == 2:
            start_d, end_d = st.session_state.active_dates
            final_df = final_df[(final_df['date'] >= start_d) & (final_df['date'] <= end_d)]

        # Render Final Feed
        st.write(f"Showing **{len(final_df)}** relevant articles.")

        if final_df.empty:
            st.warning("No articles match your current filter combination.")
        else:
            for _, row in final_df.iterrows():
                sentiment_color = "#00ffa3" if row['sentiment'] == 'POSITIVE' else "#ff4b4b" if row['sentiment'] == 'NEGATIVE' else "#888888"
                st.markdown(f"""
                    <div class="feed-card" style="border-left-color: {sentiment_color}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #888; font-size: 0.8rem;">{row['source']} • {str(row['publishedAt'])[:10]}</span>
                            <div style="color: {sentiment_color}; font-size: 0.85rem; font-weight: 800;">
                                {row['sentiment']} ({row['confidence']:.1%})
                            </div>
                        </div>
                        <div style="font-weight: 600; margin-top: 5px; color: #fff; font-size: 1rem;"><a href="{row['url']}" target="_blank" style="color: inherit; text-decoration: none;">{row['title']}</a></div>
                        <div style="font-size: 0.85rem; color: #bbb; margin-top: 5px;">{str(row['text'])[:200]}...</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; padding: 50px;"><h1 style="font-size: 3rem;">Ready for Insight?</h1><p style="color: #888;">Enter a keyword in the sidebar to generate your report.</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
