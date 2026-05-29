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
}

/* TEXT INSIDE SELECTBOX */
.stSelectbox div[data-baseweb="select"] span {
    color: #0F172A !important;
    font-weight: 600 !important;
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

/* METRIC TITLE */
[data-testid="metric-container"] label {
    color: #1E3A8A !important;
    font-weight: 800 !important;
}

/* METRIC VALUE */
[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: white !important;
    border-radius: 16px;
    padding: 15px;
    border: 2px dashed #93C5FD;
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
    "⊹ Upload Dataset",
    type=["csv", "xlsx"]
)

# ==================================================
# MAIN SYSTEM
# ==================================================

if uploaded_file is not None:

    # ==================================================
    # LOAD DATASET
    # ==================================================

    try:

        file_name = uploaded_file.name

        # CSV
        if file_name.endswith(".csv"):

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

        # EXCEL
        elif file_name.endswith(".xlsx"):

            df = pd.read_excel(uploaded_file)

        st.success("⊹ Dataset Loaded Successfully ✨")

    except Exception as e:

        st.error(f"Error loading dataset: {e}")

    # ==================================================
    # DATA PREVIEW
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
    st.sidebar.write("⊹ State of Nature : kondisi")
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
⊹ Berdasarkan metode Maximax, alternatif **{best_choice}** dipilih karena memiliki nilai payoff tertinggi dibandingkan alternatif lainnya.

⊹ Metode Maximax berfokus pada kemungkinan hasil paling menguntungkan (optimistic approach), sehingga alternatif dengan potensi keuntungan maksimum akan direkomendasikan.

⊹ Dari hasil perhitungan DSS, alternatif **{best_choice}** memperoleh skor sebesar **{best_score:.2f}**, yang menunjukkan performa terbaik pada kondisi tertentu dalam dataset.
"""

        elif method == "Maximin":

            explanation = f"""
⊹ Berdasarkan metode Maximin, alternatif **{best_choice}** dipilih karena memiliki nilai minimum terbaik dibanding alternatif lainnya.

⊹ Metode ini digunakan untuk pengambilan keputusan yang lebih aman (conservative approach), dengan mempertimbangkan kondisi terburuk dari setiap alternatif.

⊹ Hasil evaluasi menunjukkan bahwa alternatif **{best_choice}** memperoleh skor sebesar **{best_score:.2f}**, sehingga dianggap sebagai pilihan paling stabil dan minim risiko.
"""

        elif method == "Expected Value":

            explanation = f"""
⊹ Berdasarkan metode Expected Value, alternatif **{best_choice}** direkomendasikan karena menghasilkan nilai ekspektasi tertinggi.

⊹ Sistem menghitung rata-rata hasil berdasarkan probabilitas setiap state of nature yang telah dimasukkan pengguna.

⊹ Dari hasil analisis DSS, alternatif **{best_choice}** memperoleh expected value sebesar **{best_score:.2f}**, sehingga dianggap paling optimal secara keseluruhan.
"""

        elif method == "Minimax Regret":

            explanation = f"""
⊹ Berdasarkan metode Minimax Regret, alternatif **{best_choice}** dipilih karena memiliki nilai regret paling kecil dibandingkan alternatif lainnya.

⊹ Metode ini bertujuan meminimalkan potensi penyesalan (regret) akibat pengambilan keputusan yang kurang optimal pada berbagai kondisi.

⊹ Hasil analisis menunjukkan bahwa alternatif **{best_choice}** memiliki nilai regret sebesar **{best_score:.2f}**, sehingga menjadi pilihan yang paling aman untuk meminimalkan kerugian keputusan.
"""

        st.info(explanation)

        # ==================================================
        # RANKING RESULT
        # ==================================================

        st.subheader("✦ Ranking Result")

        st.dataframe(ranking_df)

        # ==================================================
        # VISUALIZATION
        # ==================================================

        st.subheader("✦ DSS Visualization")

        fig = px.bar(
            ranking_df,
            x="Alternative",
            y="Score",
            color="Score",
            template="plotly_white",
            title=f"{method} Result"
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
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "⊹ Upload CSV / Excel dataset to start DSS analysis ✨"
    )
