import os
from huggingface_hub import hf_hub_download
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from difflib import get_close_matches

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍",
    layout="wide"
)
# ----------------------------
# DOWNLOAD LARGE MODEL
# ----------------------------

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/item_similarity.pkl"):

    hf_hub_download(
        repo_id="ijghb/shopper-spectrum-models",
        filename="item_similarity.pkl",
        local_dir="models"
    )
# ----------------------------
# LOAD MODELS
# ----------------------------

@st.cache_resource
def load_models():

    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    item_sim_df = pd.read_pickle(
    "models/item_similarity.pkl"
     )
    with open("models/cluster_labels.json", "r") as f:
        cluster_labels = json.load(f)

    cluster_labels = {int(k): v for k, v in cluster_labels.items()}

    rfm = pd.read_csv("models/rfm_data.csv")

    return kmeans, scaler, item_sim_df, cluster_labels, rfm


kmeans, scaler, item_sim_df, cluster_labels, rfm = load_models()

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #2D3748;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.4);
    margin-bottom: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 8px 30px rgba(0, 229, 255, 0.15);
    border-color: #00E5FF;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: #00E5FF;
}

.metric-label {
    color: #9CA3AF;
    font-size: 15px;
    margin-top: 4px;
}

/* ── Feature cards ── */
.feature-card {
    background: linear-gradient(135deg, #1a2235, #0f1724);
    padding: 24px 20px;
    border-radius: 20px;
    border: 1px solid #2D3748;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 8px 30px rgba(99, 102, 241, 0.2);
    border-color: #6366F1;
}

.feature-icon {
    font-size: 36px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: #E5E7EB;
    margin-bottom: 6px;
}

.feature-desc {
    font-size: 14px;
    color: #9CA3AF;
    line-height: 1.5;
}

/* ── Segment badges ── */
.main-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #00E5FF, #6366F1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-title {
    color: #9CA3AF;
    font-size: 18px;
    margin-top: 6px;
}

.high {
    background: linear-gradient(135deg, #065F46, #047857);
    padding: 18px;
    border-radius: 16px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 4px 15px rgba(6, 95, 70, 0.4);
    letter-spacing: 1px;
}

.regular {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    padding: 18px;
    border-radius: 16px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 4px 15px rgba(29, 78, 216, 0.4);
    letter-spacing: 1px;
}

.risk {
    background: linear-gradient(135deg, #B45309, #D97706);
    padding: 18px;
    border-radius: 16px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 4px 15px rgba(180, 83, 9, 0.4);
    letter-spacing: 1px;
}

.occasional {
    background: linear-gradient(135deg, #374151, #4B5563);
    padding: 18px;
    border-radius: 16px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 4px 15px rgba(55, 65, 81, 0.4);
    letter-spacing: 1px;
}

/* ── Buttons ── */
button[kind="primary"] {
    border-radius: 12px !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    font-size: 17px;
    font-weight: 700;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #6366F1, #00E5FF) !important;
    color: white !important;
    border: none !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
}

/* ── Sidebar nav label ── */
.sidebar-brand {
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(90deg, #00E5FF, #6366F1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 8px 0 4px 0;
    letter-spacing: 0.5px;
}

.sidebar-tagline {
    text-align: center;
    color: #6B7280;
    font-size: 12px;
    margin-bottom: 16px;
}

/* ── Divider ── */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366F1, #00E5FF, transparent);
    border: none;
    margin: 24px 0;
    border-radius: 2px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.markdown("""
<div class='sidebar-brand'>🛍 Shopper Spectrum</div>
<div class='sidebar-tagline'>AI Retail Intelligence Platform</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🛍 Product Recommender",
        "👥 Customer Segmentation",
        "📊 Customer Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
**🏠 Home**
Project overview & stats

**🛍 Recommender**
Find similar products using AI

**👥 Segmentation**
Predict your customer type

**📊 Insights**
Deep customer analytics
""")

# ----------------------------
# HOME
# ----------------------------

if page == "🏠 Home":

    st.markdown("""
    <div class='main-title'>🛍 Shopper Spectrum</div>
    <div class='sub-title'>AI-Powered Customer Segmentation & Product Recommendation System</div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-value'>541K</div>
        <div class='metric-label'>Transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-value'>4,338</div>
        <div class='metric-label'>Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-value'>3,877</div>
        <div class='metric-label'>Products</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-value'>37</div>
        <div class='metric-label'>Countries</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Platform Features ──
    st.subheader("🚀 Platform Features")
    st.write("")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown("""
        <div class='feature-card'>
        <div class='feature-icon'>🛍</div>
        <div class='feature-title'>Product Recommender</div>
        <div class='feature-desc'>Discover similar products instantly using item-based collaborative filtering.</div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class='feature-card'>
        <div class='feature-icon'>👥</div>
        <div class='feature-title'>Customer Segmentation</div>
        <div class='feature-desc'>Identify customer value and churn risk with KMeans clustering.</div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class='feature-card'>
        <div class='feature-icon'>📊</div>
        <div class='feature-title'>Customer Insights</div>
        <div class='feature-desc'>Visualise RFM patterns and segment-level analytics at a glance.</div>
        </div>
        """, unsafe_allow_html=True)

    with f4:
        st.markdown("""
        <div class='feature-card'>
        <div class='feature-icon'>🌍</div>
        <div class='feature-title'>Global Coverage</div>
        <div class='feature-desc'>Built on 541K transactions spanning 37 countries worldwide.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Segment Distribution Pie ──
    seg_counts = rfm["Segment"].value_counts()

    fig = px.pie(
        values=seg_counts.values,
        names=seg_counts.index,
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        hole=0.4
    )

    fig.update_layout(
        template="plotly_dark",
        title_font_size=20
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# PRODUCT RECOMMENDER
# ----------------------------

elif page == "🛍 Product Recommender":

    st.title("🛍 Product Recommendation Engine")

    st.info("""
    **How to use?**

    1. 🔍 Type a keyword to search for a product
    2. 📦 Select the product from the dropdown
    3. 🚀 Click **Get Recommendations**
    4. ✨ AI will instantly suggest the 5 most similar products
    """)

    st.write("")

    search = st.text_input("🔍 Search Product", placeholder="e.g. mug, candle, bag...")

    all_products = sorted(item_sim_df.index.tolist())

    filtered_products = (
        [p for p in all_products if search.lower() in p.lower()]
        if search
        else all_products[:100]
    )

    if len(filtered_products) == 0:
        filtered_products = all_products[:100]

    product = st.selectbox("📦 Choose Product", filtered_products)

    st.write("")

    if st.button("🚀 Get Recommendations"):

        products_list = list(item_sim_df.index)

        match = get_close_matches(
            product,
            products_list,
            n=1,
            cutoff=0.3
        )

        if len(match) == 0:
            st.error("❌ Product not found. Try a different search term.")

        else:
            selected_product = match[0]

            recommendations = (
                item_sim_df[selected_product]
                .sort_values(ascending=False)
                .iloc[1:6]
            )

            st.success(f"✅ Matched Product: **{selected_product}**")
            st.write("")

            for prod, score in recommendations.items():
                st.markdown(
                    f"""
                    <div class='metric-card'>
                    <h4 style='color:#E5E7EB; margin:0 0 6px 0;'>{prod}</h4>
                    <p style='color:#00E5FF; font-size:16px; margin:0;'>
                        Similarity Score: <strong>{score:.3f}</strong>
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            fig = px.bar(
                x=recommendations.values,
                y=recommendations.index,
                orientation='h',
                title="Top 5 Similar Products",
                color=recommendations.values,
                color_continuous_scale="Teal",
                labels={"x": "Similarity Score", "y": "Product"}
            )

            fig.update_layout(
                template="plotly_dark",
                coloraxis_showscale=False,
                title_font_size=18
            )

            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SEGMENTATION
# ----------------------------

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.info("""
    **What is RFM?**

    | Metric | Meaning |
    |---|---|
    | **Recency** | How many days ago was the customer's last purchase? |
    | **Frequency** | How many times has the customer purchased? |
    | **Monetary** | How much has the customer spent in total (£)? |
    """)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        recency = st.number_input(
            "📅 Recency (days)",
            min_value=0,
            value=30,
            help="Number of days since the last purchase"
        )

    with c2:
        frequency = st.number_input(
            "🔁 Frequency",
            min_value=1,
            value=5,
            help="Total number of purchases made"
        )

    with c3:
        monetary = st.number_input(
            "💷 Monetary (£)",
            min_value=0.0,
            value=1000.0,
            help="Total amount spent by the customer"
        )

    st.write("")

    if st.button("🔮 Predict Segment"):

        sample = np.array([[recency, frequency, monetary]])
        scaled = scaler.transform(sample)
        cluster = kmeans.predict(scaled)[0]
        segment = cluster_labels[cluster]

        st.write("")
        st.subheader("Predicted Segment")

        if segment == "High-Value":
            st.markdown(
                "<div class='high'>🏆 HIGH VALUE CUSTOMER</div>",
                unsafe_allow_html=True
            )
            st.success("💎 Offer VIP rewards, early access deals, and loyalty benefits.")

        elif segment == "Regular":
            st.markdown(
                "<div class='regular'>⭐ REGULAR CUSTOMER</div>",
                unsafe_allow_html=True
            )
            st.info("📢 Increase engagement through targeted promotions and upselling.")

        elif segment == "At-Risk":
            st.markdown(
                "<div class='risk'>⚠️ AT-RISK CUSTOMER</div>",
                unsafe_allow_html=True
            )
            st.warning("🚨 Customer may churn. Launch a re-engagement / retention campaign immediately.")

        else:
            st.markdown(
                "<div class='occasional'>🛒 OCCASIONAL CUSTOMER</div>",
                unsafe_allow_html=True
            )
            st.info("🎯 Encourage repeat purchases with personalised offers and reminders.")

        st.write("")

        col_g1, col_g2, col_g3 = st.columns(3)

        gauges = [
            ("Recency (days)", recency, 365, "#00E5FF"),
            ("Frequency", frequency, 200, "#6366F1"),
            ("Monetary (£)", monetary, 50000, "#10B981"),
        ]

        for col, (label, val, max_val, color) in zip(
            [col_g1, col_g2, col_g3], gauges
        ):
            g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label, "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, max_val]},
                    "bar": {"color": color},
                    "bgcolor": "#1F2937",
                    "bordercolor": "#2D3748"
                }
            ))
            g.update_layout(template="plotly_dark", height=250, margin=dict(t=60, b=20))
            col.plotly_chart(g, use_container_width=True)

# ----------------------------
# INSIGHTS
# ----------------------------

elif page == "📊 Customer Insights":

    st.title("📊 Customer Insights")

    fig1 = px.scatter(
        rfm,
        x="Recency",
        y="Frequency",
        color="Segment",
        title="Recency vs Frequency by Segment",
        opacity=0.75,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig1.update_layout(template="plotly_dark", title_font_size=18)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Segment",
        title="Frequency vs Monetary by Segment",
        opacity=0.75,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig2.update_layout(template="plotly_dark", title_font_size=18)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    segment_stats = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
    )

    st.subheader("📋 Average RFM by Segment")
    st.dataframe(segment_stats, use_container_width=True)
