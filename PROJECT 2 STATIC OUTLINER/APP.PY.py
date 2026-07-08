import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Outlier Detection using Z-Score", layout="wide")

st.title("📊 Outlier Detection using Z-Score")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric columns found in the dataset.")
        st.stop()

    column = st.selectbox("Select a Numeric Column", numeric_cols)

    st.subheader("Statistics")
    st.write(df[column].describe())

    mean = df[column].mean()
    std = df[column].std()

    df["Z-Score"] = (df[column] - mean) / std

    threshold = st.slider(
        "Select Z-Score Threshold",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
    )

    outliers = df[np.abs(df["Z-Score"]) > threshold]

    st.subheader("Detected Outliers")
    st.write(f"Total Outliers: {len(outliers)}")
    st.dataframe(outliers)

    st.subheader("Histogram")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(df[column], bins=30)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Box Plot")

    fig2, ax2 = plt.subplots(figsize=(8,2))
    ax2.boxplot(df[column], vert=False)
    ax2.set_xlabel(column)
    st.pyplot(fig2)

    st.download_button(
        "Download Outliers",
        outliers.to_csv(index=False),
        "outliers.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")
