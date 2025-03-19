import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Configuration
st.set_page_config(page_title="📊 Project Superhuman 20290825", layout="wide")
st.title("🚀 Project Superhuman 20290825 - Life Growth Dashboard")

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

    # Ensure necessary columns exist
    required_columns = ["Development Area", "Current Level", "Target", "Improvements Needed", "Actionable Steps"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing columns: {missing_columns}. Please check your file headers.")
        st.stop()

    # 📊 **Full Overview Table (All Development Areas)**
    st.subheader("📋 Full Overview of All Development Areas")
    st.dataframe(df)

    # 🎯 **AI Insights: Predict High-Priority Areas**
    st.subheader("🤖 AI-Powered Focus Insights")

    # Encode categorical variables for AI prediction
    encoded_df = pd.get_dummies(df.drop(columns=["Development Area"]))

    # Generate dummy AI priority labels (1 = High Priority, 0 = Low Priority)
    target_labels = np.random.choice([0, 1], size=len(df))

    # Train AI Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(encoded_df, target_labels)
    predictions = rf_model.predict(encoded_df)
    df["AI_Priority"] = predictions

    # Show AI-Recommended High Priority Areas
    high_priority_areas = df[df["AI_Priority"] == 1]["Development Area"].unique()
    st.write("🔍 **AI suggests focusing on these areas first:**")
    st.write(high_priority_areas)

    # 📊 **Visualizing Personal Growth**
    st.subheader("📈 Progress Across All Development Areas")

    # 📍 Progress Bar for Each Area
    for index, row in df.iterrows():
        st.write(f"**{row['Development Area']}**")
        progress_value = min(100, (index + 1) / len(df) * 100)
        st.progress(progress_value / 100)
    
    # 📊 **Comparison of Current vs Target Levels**
    st.subheader("🎯 Progress Towards Target Levels")
    fig = px.bar(df, x="Development Area", y=["Current Level", "Target"], barmode="group", title="Current vs Target Levels")
    st.plotly_chart(fig, use_container_width=True)

    # 📊 **Improvements Needed Distribution**
    st.subheader("⚡ Improvements Needed Across Areas")
    fig2 = px.histogram(df, x="Improvements Needed", title="Distribution of Required Improvements")
    st.plotly_chart(fig2, use_container_width=True)

    # ✅ **Actionable Steps for Each Area**
    st.subheader("📝 Personalized Action Plan")
    for _, row in df.iterrows():
        with st.expander(f"📌 {row['Development Area']}"):
            st.write(f"**Improvements Needed:** {row['Improvements Needed']}")
            st.write(f"**Actionable Steps:** {row['Actionable Steps']}")

