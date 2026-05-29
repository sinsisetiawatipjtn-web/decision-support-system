import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Decision Support System",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-color: #F8FAFC;
}

/* FONT */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #0F172A;
}

/* TITLE */
h1 {
    color: #0F172A !important;
    font-size: 52px !important;
    font-weight: 800 !important;
}

/* SUBTITLE */
h2, h3 {
    color: #1E3A8A !important;
    font-weight: 700 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    padding: 20px;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* SELECTBOX */
.stSelectbox div[data-baseweb="select"] {
    background-color: white;
    border-radius: 10px;
    color: black !important;
}

/* MULTISELECT */
.stMultiSelect div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px;
}

.stMultiSelect span {
    color: black !important;
}

/* BUTTON */
.stButton>button {
    background-color: #2563EB;
    color: white !important;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* ALERT */
div.stAlert {
    border-radius: 15px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #CBD5E1;
}

/* METRIC BOX */
[data-testid="metric-container"] {
    background-color: white !important;
    border: 1px solid #CBD5E1 !important;
    padding: 20px !important;
    border-radius: 16px !important;
}

/* JUDUL METRIC */
[data-testid="metric-container"] label {
    color: #1E3A8A !important;
    font-weight: 800 !important;
}

/* ANGKA/TEXT METRIC */
[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

/* METHOD / DELTA */
[data-testid="stMetricDelta"] {
    color: #2563EB !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border-radius: 16px;
    padding: 15px;
    border: 2px dashed #93C5FD;
}

/* FILE UPLOADER TEXT */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label {
    color: #0F172A !important;
}

/* INFO BOX */
[data-testid="stAlert"] {
    color: #0F172A !important;
}

/* PLOT TEXT */
.js-plotly-plot .plotly text {
    fill: #334155 !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.title("✦ Decision Support System ✦")

st.markdown("""
<h3 style='color:#1E3A8A; margin-bottom:10px;'>
⊹ Smart Decision Theory Dashboard
</h3>

<div style="
background-color:#E2E8F0;
padding:14px 18px;
border-radius:14px;
border-left:6px solid #2563EB;
font-size:17px;
font-weight:500;
color:#334155;
box-shadow:0 2px 8px rgba(0,0,0,0.05);
width:fit-content;
margin-bottom:20px;
">
✧ Upload dataset → choose DSS method → get automatic recommendation ✨
</div>
""", unsafe_allow_html=True)

st.write("---")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "⊹ Upload CSV Dataset",
    type=["csv"]
)

# ==================================================
# MAIN SYSTEM
# ==================================================

if uploaded_file is not None:

    # ==================================================
    # LOAD DATASET
    # ==================================================

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="utf-8"
        )

    except:

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                encoding="latin1"
            )

        except:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                encoding="ISO-8859-1"
            )

    st.success("⊹ Dataset Loaded Successfully ✨")

    # ==================================================
    # PREVIEW
    # ==================================================

    st.subheader("✦ Dataset Preview")

    st.dataframe(df.head())

    # ==================================================
    # COLUMN DETECTION
    # ==================================================

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    text_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # ==================================================
    # SIDEBAR
    # ==================================================

    st.sidebar.title("✦ DSS Configuration")

    alternative_col = st.sidebar.selectbox(
        "✦ Choose Alternative",
        text_cols
    )

    payoff_cols = st.sidebar.multiselect(
        "✦ Choose State of Nature",
        numeric_cols,
        default=numeric_cols[:3]
    )

    method = st.sidebar.selectbox(
        "✦ Choose DSS Method",
        [
            "Maximax",
            "Maximin",
            "Expected Value",
            "Minimax Regret"
        ]
    )

    st.sidebar.write("---")

    st.sidebar.subheader("✦ DSS Components")

    st.sidebar.write("⊹ Alternative : opsi keputusan")
    st.sidebar.write("⊹ State of Nature : kondisi/variabel")
    st.sidebar.write("⊹ Payoff : hasil keputusan")
    st.sidebar.write("⊹ Method : metode DSS")

    # ==================================================
    # PAYOFF TABLE
    # ==================================================

    if len(payoff_cols) > 0:

        payoff_table = df[
            [alternative_col] + payoff_cols
        ]

        payoff_table = payoff_table.groupby(
            alternative_col
        ).mean()

        st.subheader("✦ Payoff Table")

        st.dataframe(payoff_table)

        # ==================================================
        # MAXIMAX
        # ==================================================

        if method == "Maximax":

            result = payoff_table.max(axis=1)

            best_choice = result.idxmax()

            best_score = result.max()

        # ==================================================
        # MAXIMIN
        # ==================================================

        elif method == "Maximin":

            result = payoff_table.min(axis=1)

            best_choice = result.idxmax()

            best_score = result.max()

        # ==================================================
        # EXPECTED VALUE
        # ==================================================

        elif method == "Expected Value":

            st.subheader("✦ Input State Probabilities")

            probabilities = []

            total_prob = 0

            cols = st.columns(len(payoff_cols))

            for i, col in enumerate(payoff_cols):

                with cols[i]:

                    p = st.number_input(
                        f"{col}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1
                    )

                    probabilities.append(p)

                    total_prob += p

            if total_prob == 1:

                prob_array = np.array(probabilities)

                result = payoff_table.dot(prob_array)

                best_choice = result.idxmax()

                best_score = result.max()

            else:

                st.warning(
                    "⚠ Total probability must equal 1"
                )

                st.stop()

        # ==================================================
        # MINIMAX REGRET
        # ==================================================

        elif method == "Minimax Regret":

            regret_table = (
                payoff_table.max() - payoff_table
            )

            st.subheader("✦ Regret Table")

            st.dataframe(regret_table)

            result = regret_table.max(axis=1)

            best_choice = result.idxmin()

            best_score = result.min()

        # ==================================================
        # RANKING
        # ==================================================

        ranking_df = pd.DataFrame({

            "Alternative": result.index,
            "Score": result.values

        })

        if method == "Minimax Regret":

            ranking_df = ranking_df.sort_values(
                by="Score"
            )

        else:

            ranking_df = ranking_df.sort_values(
                by="Score",
                ascending=False
            )

        # ==================================================
        # METRICS
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "✦ Best Decision",
                best_choice
            )

        with col2:
            st.metric(
                "✦ Best Score",
                f"{best_score:.2f}"
            )

        with col3:
            st.metric(
                "✦ Method",
                method
            )

        # ==================================================
        # EXPLANATION
        # ==================================================

        st.subheader("✦ Recommendation Insight")

        if method == "Maximax":

            explanation = f"""
            ⊹ {best_choice} dipilih karena memiliki payoff maksimum tertinggi.
            """

        elif method == "Maximin":

            explanation = f"""
            ⊹ {best_choice} dipilih karena memiliki nilai minimum terbaik.
            """

        elif method == "Expected Value":

            explanation = f"""
            ⊹ {best_choice} dipilih karena memiliki expected value tertinggi.
            """

        elif method == "Minimax Regret":

            explanation = f"""
            ⊹ {best_choice} dipilih karena memiliki regret minimum.
            """

        st.info(f"""

        ⊹ Berdasarkan analisis menggunakan metode **{method}**, 
        alternatif **{best_choice}** direkomendasikan sebagai pilihan terbaik 
        karena menghasilkan nilai payoff paling optimal dibandingkan alternatif lainnya.

        ⊹ Sistem melakukan evaluasi terhadap seluruh state of nature 
        yang dipilih pada dataset dan menghitung performa setiap alternatif 
        berdasarkan metode pengambilan keputusan DSS.

        ⊹ Hasil akhir menunjukkan bahwa **{best_choice}** memiliki 
        skor sebesar **{best_score:.2f}**, sehingga alternatif ini dianggap 
        paling sesuai untuk dijadikan rekomendasi keputusan.

        """)

        # ==================================================
        # RANKING RESULT
        # ==================================================

        st.subheader("✦ Ranking Result")

        st.dataframe(ranking_df)

        # ==================================================
        # VISUALIZATION
        # ==================================================

        st.subheader("✦ DSS Visualization")

        fig = px.histogram(
            ranking_df,
            x="Score",
            nbins=20,
            template="plotly_white",
            title=f"{method} Score Distribution"
        )

        fig.update_layout(

            paper_bgcolor="white",
            plot_bgcolor="white",

            font=dict(
                color="#0F172A",
                size=14
            ),

            title_font=dict(
                size=22,
                color="#1E3A8A"
            ),

            xaxis_title="Score",
            yaxis_title="Frequency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "⊹ Upload CSV dataset to start DSS analysis ✨"
    )