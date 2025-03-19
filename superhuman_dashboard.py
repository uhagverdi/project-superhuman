import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Configuration
st.set_page_config(page_title="Project Superhuman 20290825 Dashboard", layout="wide")
st.title("🚀 Project Superhuman 20290825 Dashboard")

# File Uploader to Load Google Sheets File
uploaded_file = st.file_uploader("📂 Upload Your Development Plan (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Determine file type and read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # 🔹 Fix column names (remove extra spaces and standardize)
    df.columns = df.columns.str.strip().str.title()
    st.write("🔍 Final Column Names →", list(df.columns))

    # Ensure "Development Area" column exists
    if "Development Area" not in df.columns:
        st.error("❌ Error: 'Development Area' column not found! Check your file headers.")
        st.stop()

    # Sidebar Filters
    st.sidebar.header("📊 Filter Development Areas")
    selected_area = st.sidebar.selectbox("Select Development Area", df["Development Area"].unique())

    # Filter Data Based on Selection
    df_filtered = df[df["Development Area"] == selected_area]

    # Progress Overview Metrics
    st.subheader("📊 Personal Development Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Level", df_filtered.iloc[0]["Current Level"])
    col2.metric("Target Level", df_filtered.iloc[0]["Target"])
    col3.metric("Improvements Needed", df_filtered.iloc[0]["Improvements Needed"])

    # Progress Bar for Development Areas
    progress = len(df_filtered) / len(df) * 100
    st.progress(progress / 100)
    st.write(f"📈 **Your Progress in {selected_area}:** {progress:.2f}%")

    # 🔹 AI Insights: Predict High-Priority Areas
    st.subheader("🤖 AI Insights: Key Focus Areas")

    # Encode categorical variables for AI prediction
    encoded_df = pd.get_dummies(df.drop(columns=["Development Area"]))

    # Generate dummy AI priority labels (1 = High Priority, 0 = Low Priority)
    target_labels = np.random.choice([0, 1], size=len(df))

    # Train AI Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(encoded_df, target_labels)
    predictions = rf_model.predict(encoded_df)
    df["AI_Priority"] = predictions

    # Show AI Recommendations
    high_priority_areas = df[df["AI_Priority"] == 1]["Development Area"].unique()
    st.write("🔍 **AI suggests focusing on these areas first:**")
    st.write(high_priority_areas)

    # 📊 **Visualizing Personal Growth**
    st.subheader("📈 Progress & Growth Insights")
    fig = px.bar(df_filtered, x="Development Area", y="Current Level", color="Target", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # ✅ **Personalized Actionable Steps**
    st.subheader("✅ Actionable Steps for Growth")
    st.write(df_filtered.iloc[0]["Actionable Steps"])
