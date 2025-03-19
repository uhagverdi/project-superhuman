import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Configuration
st.set_page_config(page_title="Project Superhuman 20290825 Dashboard", layout="wide")
st.title("🚀 Project Superhuman 20290825 Dashboard")

# File Uploader to Load Google Sheets File
uploaded_file = st.file_uploader("Upload Your Google Sheet (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Determine file type and read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")
    
    # 🔍 Debugging: Print Column Names in Streamlit
    st.write("🔍 Debugging: File Columns Found →", list(df.columns))  

    # **Check if "Development Area" Exists Before Using It**
    if "Development Area" not in df.columns:
        st.error("❌ Error: 'Development Area' column not found! Check your file headers.")
        st.stop()  # Stop execution here if column is missing

    # Now you can safely use "Development Area"
    st.sidebar.header("Filter Development Areas")
    selected_area = st.sidebar.selectbox("Select Development Area", df["Development Area"].unique())
    df_filtered = df[df["Development Area"] == selected_area]
    
    # Display Progress Metrics
    st.subheader("📊 Progress Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Level", df_filtered.iloc[0]["Current Level"])
    col2.metric("Target Level", df_filtered.iloc[0]["Target"])
    col3.metric("Improvement Needed", df_filtered.iloc[0]["Improvements Needed"])
    
    # Progress Bar for Development Areas
    progress = len(df_filtered) / len(df) * 100
    st.progress(progress / 100)
    st.write(f"Current Progress in {selected_area}: {progress:.2f}%")
    
    # AI Insights - Predict Focus Areas
    st.subheader("🤖 AI Insights: Priority Areas")
    
    # Dummy Data Preparation for AI Model
    encoded_df = pd.get_dummies(df.drop(columns=["Development Area"]))
    target_labels = np.random.choice([0, 1], size=len(df))  # 0 = Low Priority, 1 = High Priority
    
    # Train AI Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(encoded_df, target_labels)
    predictions = rf_model.predict(encoded_df)
    df["AI_Priority"] = predictions
    
    # Show AI Recommendations
    high_priority_areas = df[df["AI_Priority"] == 1]["Development Area"].unique()
    st.write("🔍 AI suggests focusing on these areas first:")
    st.write(high_priority_areas)
    
    # Visualizations
    st.subheader("📈 Visualizing Growth")
    fig = px.bar(df_filtered, x="Development Area", y="Current Level", color="Target", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    # Actionable Steps
    st.subheader("✅ Actionable Steps")
    st.write(df_filtered.iloc[0]["Actionable Steps"])
