import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re, string
import warnings
warnings.filterwarnings('ignore')

vader = SentimentIntensityAnalyzer()

# ── Train LR model on dataset ─────────────────
@st.cache_resource
def build_lr_model():
    import pickle
    with open('lr_model.pkl', 'rb') as f:
        return pickle.load(f)
    
lr_model = build_lr_model()

def clean_text(t):
    t = t.lower()
    t = re.sub(r'http\S+|@\w+', '', t)
    return t.translate(str.maketrans('', '', string.punctuation)).strip()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0a0f1e 50%, #0d1117 100%);
    color: #e8eaf0;
}

/* FIX 1: Remove the large blank header space at top */
header[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    visibility: hidden !important;
}
/* Also remove the toolbar/deploy button area gap */
[data-testid="stToolbar"] {
    display: none !important;
}

/* FIX 2: Tighten block container top padding & make fully responsive */
.block-container {
    padding-top: 1rem !important;
    padding-left: clamp(0.75rem, 3vw, 2rem) !important;
    padding-right: clamp(0.75rem, 3vw, 2rem) !important;
    max-width: 100% !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,180,255,0.15);
}
[data-testid="stSidebar"] * {
    color: #c8d8f0 !important;
}

/* FIX 3: Hand/pointer cursor on filter sidebar elements */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSelectbox [role="combobox"],
[data-testid="stSidebar"] label {
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    border-color: rgba(0,180,255,0.55) !important;
    box-shadow: 0 0 0 2px rgba(0,180,255,0.15) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
/* Dropdown option list items */
[data-testid="stSidebar"] ul[role="listbox"] li {
    cursor: pointer !important;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2040 60%, #061830 100%);
    border: 1px solid rgba(0,180,255,0.2);
    border-radius: 20px;
    padding: clamp(18px, 3vw, 30px) clamp(18px, 4vw, 32px);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 48px rgba(0,100,255,0.12);
}
.hero::before {
    content: '✈';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: clamp(48px, 7vw, 84px);
    opacity: 0.08;
    line-height: 1;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(18px, 3.5vw, 36px) !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 10px !important;
    letter-spacing: -0.5px;
    line-height: 1.15 !important;
}
.hero p {
    color: #7ea8d0 !important;
    font-size: clamp(12px, 1.4vw, 15px) !important;
    margin: 0 !important;
    font-weight: 300;
    max-width: 85%;
}
.hero .badge {
    display: inline-block;
    background: rgba(0,180,255,0.15);
    border: 1px solid rgba(0,180,255,0.3);
    color: #4dc8ff !important;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* Metric cards — responsive grid */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: clamp(10px, 1.5vw, 16px);
    margin-bottom: 24px;
}
@media (max-width: 900px) {
    .metric-row { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
    .metric-row { grid-template-columns: 1fr 1fr; gap: 8px; }
    .hero::before { display: none; }
}
.metric-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border-radius: 16px;
    padding: clamp(14px, 2vw, 20px) clamp(10px, 1.5vw, 18px);
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .val {
    font-family: 'Syne', sans-serif;
    font-size: clamp(20px, 2.8vw, 30px);
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .lbl {
    font-size: clamp(9px, 0.9vw, 11px);
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.6;
    font-weight: 500;
}
.neg  { color: #ff5f6d; }
.neu  { color: #a0aec0; }
.pos  { color: #43e97b; }
.tot  { color: #4dc8ff; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(14px, 1.6vw, 18px);
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* Chart card */
.chart-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: clamp(14px, 2vw, 22px);
    margin-bottom: 20px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.25);
}

/* Live analyzer */
.analyzer-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid rgba(0,180,255,0.2);
    border-radius: 20px;
    padding: clamp(16px, 2.5vw, 28px) clamp(14px, 2vw, 24px);
    margin-bottom: 24px;
    box-shadow: 0 8px 40px rgba(0,100,255,0.1);
}
.result-positive {
    background: linear-gradient(135deg, rgba(67,233,123,0.12), rgba(67,233,123,0.05));
    border: 1px solid rgba(67,233,123,0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(255,95,109,0.12), rgba(255,95,109,0.05));
    border: 1px solid rgba(255,95,109,0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.result-neutral {
    background: linear-gradient(135deg, rgba(160,174,192,0.12), rgba(160,174,192,0.05));
    border: 1px solid rgba(160,174,192,0.3);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.result-emoji { font-size: 40px; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin: 8px 0 4px;
}
.result-score { font-size: 13px; opacity: 0.7; }

/* Insight box */
.insight-box {
    background: rgba(0,180,255,0.07);
    border-left: 3px solid #4dc8ff;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #c8d8f0;
    line-height: 1.6;
}

/* FIX 4: Text area — dark, readable text instead of white */
.stTextArea textarea {
    background: #f2f5f9 !important;
    border: 1px solid rgba(0,180,255,0.3) !important;
    border-radius: 12px !important;
    color: #1a2233 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    caret-color: #0057ff !important;
}
.stTextArea textarea::placeholder {
    color: #8a9dbf !important;
    opacity: 1 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,180,255,0.65) !important;
    box-shadow: 0 0 0 3px rgba(0,180,255,0.12) !important;
    background: #ffffff !important;
    color: #1a2233 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0057ff, #0099ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(0,100,255,0.35) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,100,255,0.5) !important;
}

/* Selectboxes */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,180,255,0.2) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    cursor: pointer !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(0,180,255,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,180,255,0.1) !important;
}

div[data-testid="stMarkdownContainer"] p { color: #c8d8f0; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #7ea8d0 !important;
    cursor: pointer !important;
    font-size: clamp(11px, 1.3vw, 14px) !important;
}
.stTabs [aria-selected="true"] { color: #4dc8ff !important; }

footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
DARK_BG   = '#0d1f3c'
CARD_BG   = '#0a1628'
GRID_CLR  = '#1a2f50'
TEXT_CLR  = '#c8d8f0'
NEG_CLR   = '#ff5f6d'
NEU_CLR   = '#a0aec0'
POS_CLR   = '#43e97b'
ACC_CLR   = '#4dc8ff'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   DARK_BG,
    'axes.edgecolor':   GRID_CLR,
    'axes.labelcolor':  TEXT_CLR,
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      TEXT_CLR,
    'ytick.color':      TEXT_CLR,
    'grid.color':       GRID_CLR,
    'grid.alpha':       0.5,
    'text.color':       TEXT_CLR,
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.titlepad':    14,
    'axes.labelsize':   11,
    'font.family':      'DejaVu Sans',
})

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Tweets.csv')
    df = df[['airline_sentiment','negativereason','airline','text','tweet_created']]
    df = df.dropna(subset=['text'])
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], utc=True, errors='coerce')
    return df

try:
    df = load_data()
    data_loaded = True
except:
    data_loaded = False

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:42px;'>✈️</div>
        <div style='font-family:Syne,sans-serif; font-size:17px; font-weight:800; color:#fff; margin-top:8px;'>SentimentAir</div>
        <div style='font-size:11px; color:#4dc8ff; letter-spacing:1.5px; text-transform:uppercase; margin-top:4px;'>Analytics Dashboard</div>
    </div>
    <hr style='border-color:rgba(0,180,255,0.15); margin-bottom:20px;'>
    """, unsafe_allow_html=True)

    st.markdown("**🔍 Filter Data**")
    if data_loaded:
        airlines = ['All Airlines'] + sorted(df['airline'].dropna().unique().tolist())
        selected_airline = st.selectbox("Select Airline", airlines)

        sentiments = ['All Sentiments', 'negative', 'neutral', 'positive']
        selected_sentiment = st.selectbox("Select Sentiment", sentiments)

        # Apply filters
        filtered_df = df.copy()
        if selected_airline != 'All Airlines':
            filtered_df = filtered_df[filtered_df['airline'] == selected_airline]
        if selected_sentiment != 'All Sentiments':
            filtered_df = filtered_df[filtered_df['airline_sentiment'] == selected_sentiment]
    else:
        st.warning("Tweets.csv not found")
        filtered_df = pd.DataFrame()

    st.markdown("<hr style='border-color:rgba(0,180,255,0.1); margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#4a6a90; line-height:1.7;'>
    📊 Dataset: US Airline Tweets<br>
    🐦 14,640 tweets analysed<br>
    🛠 Built with Python & Streamlit<br>
    👨‍💻 B.Tech AI & Data Science
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">Live Analytics Dashboard</div>
    <h1>✈️ Airline Sentiment Analysis</h1>
    <p>Analyzing 14,640 tweets to uncover what passengers truly feel about US Airlines — powered by NLP & Python.</p>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("⚠️ Could not load Tweets.csv — make sure it's in the same folder as app.py")
    st.stop()

# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
counts = filtered_df['airline_sentiment'].value_counts()
neg_c = counts.get('negative', 0)
neu_c = counts.get('neutral', 0)
pos_c = counts.get('positive', 0)
tot_c = len(filtered_df)

neg_pct = round(neg_c / tot_c * 100, 1) if tot_c else 0
pos_pct = round(pos_c / tot_c * 100, 1) if tot_c else 0

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="val tot">{tot_c:,}</div>
        <div class="lbl">Total Tweets</div>
    </div>
    <div class="metric-card">
        <div class="val neg">{neg_c:,}</div>
        <div class="lbl">Negative · {neg_pct}%</div>
    </div>
    <div class="metric-card">
        <div class="val neu">{neu_c:,}</div>
        <div class="lbl">Neutral</div>
    </div>
    <div class="metric-card">
        <div class="val pos">{pos_c:,}</div>
        <div class="lbl">Positive · {pos_pct}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "✈️ By Airline", "🔴 Complaints", "🤖 Live Analyzer"])

# ── TAB 1: OVERVIEW ──────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ['Negative', 'Neutral', 'Positive'],
            [neg_c, neu_c, pos_c],
            color=[NEG_CLR, NEU_CLR, POS_CLR],
            edgecolor='none',
            width=0.55
        )
        for bar, val in zip(bars, [neg_c, neu_c, pos_c]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Overall Sentiment Breakdown')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sentiment Share (Pie)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            [neg_c, neu_c, pos_c],
            labels=['Negative', 'Neutral', 'Positive'],
            colors=[NEG_CLR, NEU_CLR, POS_CLR],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2.5},
            pctdistance=0.78
        )
        for t in texts:
            t.set_color(TEXT_CLR)
            t.set_fontsize(11)
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax.set_title('Proportion of Sentiments')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    💡 <strong>Key Insight:</strong> {neg_pct}% of tweets about airlines are negative.
    This indicates a serious customer satisfaction issue across US carriers.
    Only {pos_pct}% of passengers had a positive experience worth tweeting about.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2: BY AIRLINE ────────────────────────
with tab2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sentiment Breakdown by Airline</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    airline_sent = filtered_df.groupby(['airline','airline_sentiment']).size().unstack(fill_value=0)
    airlines_list = airline_sent.index.tolist()
    x = range(len(airlines_list))
    w = 0.25
    for i, (col, clr) in enumerate(zip(['negative','neutral','positive'], [NEG_CLR, NEU_CLR, POS_CLR])):
        if col in airline_sent.columns:
            vals = airline_sent[col].values
            bars = ax.bar([xi + i*w for xi in x], vals, width=w, color=clr,
                         edgecolor='none', label=col.capitalize())
    ax.set_xticks([xi + w for xi in x])
    ax.set_xticklabels(airlines_list, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Number of Tweets')
    ax.set_title('Sentiment Comparison Across Airlines')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(framealpha=0, labelcolor=TEXT_CLR, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Key Insight:</strong> United Airlines receives the highest number of negative tweets,
    followed closely by US Airways and American Airlines. Virgin America has the
    best positive-to-negative ratio among all carriers.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 3: COMPLAINTS ────────────────────────
with tab3:
    col1, col2 = st.columns([1,1], gap="medium")

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Complaint Reasons</div>', unsafe_allow_html=True)
        neg_df = filtered_df[filtered_df['airline_sentiment']=='negative']
        reason_counts = neg_df['negativereason'].value_counts().dropna().head(10)
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.barh(reason_counts.index[::-1], reason_counts.values[::-1],
                       color=[NEG_CLR]*len(reason_counts), edgecolor='none', height=0.65)
        for bar, val in zip(bars, reason_counts.values[::-1]):
            ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9, color=TEXT_CLR)
        ax.set_xlabel('Number of Tweets')
        ax.set_title('Why Are Passengers Angry?')
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Negative Tweet Word Cloud</div>', unsafe_allow_html=True)
        neg_text = ' '.join(neg_df['text'].dropna().astype(str).tolist())
        stopwords_extra = {'https', 'http', 'co', 't', 'the', 'to', 'a', 'i',
                           'and', 'is', 'in', 'it', 'of', 'for', 'on', 'my',
                           'you', 'we', 'are', 'be', 'at', 'this', 'have',
                           'so', 'but', 'an', 'me', 'RT', 'amp'}
        wc = WordCloud(
            width=700, height=380,
            background_color=DARK_BG,
            colormap='RdYlGn_r',
            max_words=120,
            stopwords=stopwords_extra,
            prefer_horizontal=0.85,
            margin=4
        ).generate(neg_text)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Common Words in Complaints')
        fig.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Key Insight:</strong> Customer Service Issues account for the largest share of complaints (2,900+ tweets),
    followed by Late Flights and Cancelled Flights. Airlines should urgently invest in
    customer service training and operational punctuality to reduce negative sentiment.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4: LIVE ANALYZER ─────────────────────
def sentiment_emoji(label):
    return "😊" if label == "Positive" else ("😠" if label == "Negative" else "😐")

def sentiment_color(label):
    return POS_CLR if label == "Positive" else (NEG_CLR if label == "Negative" else NEU_CLR)

def prob_bar_html(label, pct, clr):
    return f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
        <div style="width:70px; font-size:12px; color:#7ea8d0; flex-shrink:0;">{label}</div>
        <div style="flex:1; background:rgba(255,255,255,0.06); border-radius:6px; height:10px; overflow:hidden;">
            <div style="width:{pct:.1f}%; background:{clr}; height:100%; border-radius:6px; transition:width 0.4s;"></div>
        </div>
        <div style="width:46px; text-align:right; font-size:12px; font-weight:700; color:{clr};">{pct:.1f}%</div>
    </div>"""

def model_card_html(model_name, label, neg_pct, neu_pct, pos_pct):
    emoji = sentiment_emoji(label)
    clr   = sentiment_color(label)
    bars  = (prob_bar_html("Negative", neg_pct, NEG_CLR) +
             prob_bar_html("Neutral",  neu_pct, NEU_CLR) +
             prob_bar_html("Positive", pos_pct, POS_CLR))
    return f"""
    <div style="background:linear-gradient(135deg,#0d1f3c,#0a1628);
                border:1px solid rgba(255,255,255,0.08); border-radius:16px;
                padding:22px 20px; height:100%;">
        <div style="display:inline-block; border:1px solid rgba(255,255,255,0.2);
                    border-radius:20px; padding:3px 14px; font-size:10px;
                    letter-spacing:2px; text-transform:uppercase; color:#a0b4cc;
                    margin-bottom:18px;">{model_name}</div>
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:36px;">{emoji}</div>
            <div style="font-family:'Syne',sans-serif; font-size:22px;
                        font-weight:800; color:{clr}; margin-top:6px;">{label}</div>
        </div>
        {bars}
    </div>"""

def find_text_column(columns):
    preferred_columns = ["text", "tweet", "tweet_text", "content", "message"]
    normalized_columns = {str(column).strip().lower(): column for column in columns}
    for preferred_column in preferred_columns:
        if preferred_column in normalized_columns:
            return normalized_columns[preferred_column]
    return None

def predict_export_sentiments(source_df, text_column):
    result_df = source_df.copy()
    text_values = result_df[text_column].fillna("").astype(str)
    cleaned_values = text_values.map(clean_text).tolist()

    lr_probs = lr_model.predict_proba(cleaned_values)
    lr_classes = list(lr_model.classes_)
    result_df["lr_sentiment"] = [
        str(lr_classes[row.argmax()]).capitalize() for row in lr_probs
    ]
    result_df["lr_confidence"] = [round(float(row.max()), 4) for row in lr_probs]

    vader_scores = [vader.polarity_scores(text) for text in text_values]
    result_df["vader_sentiment"] = [
        "Positive" if score["compound"] >= 0.05 else (
            "Negative" if score["compound"] <= -0.05 else "Neutral"
        )
        for score in vader_scores
    ]
    result_df["vader_compound"] = [
        round(float(score["compound"]), 4) for score in vader_scores
    ]
    return result_df

with tab4:
    st.markdown('<div class="analyzer-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Live Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7ea8d0; font-size:14px; margin-bottom:18px;">Compare how <strong style="color:#4dc8ff;">Logistic Regression</strong> and <strong style="color:#f6c90e;">VADER Lexicon</strong> predict sentiment on the same tweet.</p>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="",
        placeholder='e.g. "My flight was delayed for 3 hours and no one helped me at the counter..."',
        height=100,
        label_visibility="collapsed"
    )

    col_btn, col_ex = st.columns([1, 2])
    with col_btn:
        analyze_btn = st.button("⚡ Analyze Sentiment")
    with col_ex:
        st.markdown('<div style="padding-top:10px; font-size:12px; color:#4a6a90;">Try: "The crew was amazing!" · "Flight cancelled again" · "Average experience"</div>', unsafe_allow_html=True)

    if analyze_btn and user_input.strip():
        cleaned = clean_text(user_input)

        # ── Logistic Regression ──
        if lr_model:
            lr_probs  = lr_model.predict_proba([cleaned])[0]
            lr_classes = list(lr_model.classes_)
            lr_neg = lr_probs[lr_classes.index('negative')] * 100
            lr_neu = lr_probs[lr_classes.index('neutral')]  * 100
            lr_pos = lr_probs[lr_classes.index('positive')] * 100
            lr_label = lr_classes[lr_probs.argmax()].capitalize()
        else:
            lr_neg, lr_neu, lr_pos, lr_label = 33.3, 33.3, 33.3, "Neutral"

        # ── VADER ──
        vs = vader.polarity_scores(user_input)
        total = vs['neg'] + vs['neu'] + vs['pos'] or 1
        vd_neg = vs['neg'] / total * 100
        vd_neu = vs['neu'] / total * 100
        vd_pos = vs['pos'] / total * 100
        compound = vs['compound']
        vd_label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(model_card_html("Logistic Regression", lr_label, lr_neg, lr_neu, lr_pos), unsafe_allow_html=True)
        with c2:
            st.markdown(model_card_html("VADER Lexicon", vd_label, vd_neg, vd_neu, vd_pos), unsafe_allow_html=True)

        agree_txt = "✅ Both models agree!" if lr_label == vd_label else "⚠️ Models disagree — results may vary by method."
        agree_clr = "#43e97b" if lr_label == vd_label else "#f6c90e"
        st.markdown(f"""
        <div style="margin-top:16px; padding:12px 16px; background:rgba(0,180,255,0.07);
                    border-left:3px solid {agree_clr}; border-radius:0 10px 10px 0;
                    font-size:13px; color:{agree_clr};">{agree_txt}</div>
        <div style="margin-top:12px; font-size:12px; color:#4a6a90;">
            ✏️ <strong style="color:#7ea8d0;">Cleaned Text:</strong>
            <em style="color:#c8d8f0;">"{cleaned}"</em>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

    st.markdown('<div class="section-title">Batch CSV Analyzer</div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader(
        "Upload a tweet CSV with a text column",
        type=["csv"],
        key="tweet_export_csv",
    )

    if uploaded_csv is not None:
        try:
            export_df = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
        else:
            text_column = find_text_column(export_df.columns)
            if text_column is None:
                st.warning("Add a text, tweet, tweet_text, content, or message column.")
            else:
                non_empty_df = export_df.dropna(subset=[text_column])
                if non_empty_df.empty:
                    st.warning("No text rows found in the uploaded CSV.")
                else:
                    predictions_df = predict_export_sentiments(non_empty_df, text_column)
                    preview_columns = [
                        text_column,
                        "lr_sentiment",
                        "lr_confidence",
                        "vader_sentiment",
                        "vader_compound",
                    ]
                    st.dataframe(predictions_df[preview_columns].head(50), use_container_width=True)
                    st.download_button(
                        "Download Scored CSV",
                        data=predictions_df.to_csv(index=False),
                        file_name="airline_tweet_sentiment_predictions.csv",
                        mime="text/csv",
                    )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 10px; color:#2a4060; font-size:12px; letter-spacing:0.5px;'>
    Built with ❤️ using Python · Pandas · Matplotlib · TextBlob · Streamlit<br>
    <span style='color:#1a3050;'>B.Tech AI & Data Science — Internship Project</span>
</div>
""", unsafe_allow_html=True)
