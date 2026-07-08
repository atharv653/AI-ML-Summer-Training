import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title and creator credit
st.title("📱 Google Play Store Data Explorer")
st.caption("Created by Atharv Singh")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Google Play Store CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # Sidebar filters
    st.sidebar.header("Filters")
    category = st.sidebar.selectbox("Select Category", df['Category'].unique())
    df_filtered = df[df['Category'] == category]

    # Show filtered data
    st.subheader(f"Apps in Category: {category}")
    st.write(df_filtered[['App', 'Rating', 'Installs']].head(10))

    # Plot ratings distribution
    st.subheader("📊 Ratings Distribution")
    fig, ax = plt.subplots()
    df_filtered['Rating'].dropna().hist(bins=20, ax=ax, color="skyblue", edgecolor="black")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Apps")
    st.pyplot(fig)

    # Top apps by installs
    st.subheader("🔥 Top 10 Apps by Installs")
    top_installs = df_filtered.sort_values("Installs", ascending=False).head(10)
    st.bar_chart(top_installs.set_index("App")["Installs"])

    # Footer credit
    st.markdown("---")
    st.markdown("**Developed with ❤️ by Atharv Singh**")

else:
    st.info("Please upload a Google Play Store dataset CSV to begin.")
