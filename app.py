import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("🏆 2026 FIFA World Cup AI Predictor")
st.markdown("This dashboard uses a Random Forest Machine Learning model to predict the probability of national teams reaching the Quarter-Finals in 2026, based on their performance over the last four years.")

# 2. Load the Data
# Ensure the CSVs are in the same folder as this app.py script
@st.cache_data
def load_data():
    predictions = pd.read_csv('2026_WC_Predictions.csv')
    importance = pd.read_csv('WC_Feature_Importance.csv')
    return predictions, importance

df_preds, df_importance = load_data()

# 3. Create Dashboard Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Top 8 Quarter-Finalist Predictions")
    # Format the probability as a percentage for better readability
    df_preds['Probability (%)'] = (df_preds['qf_probability'] * 100).round(1)
    
    # Display an interactive dataframe
    st.dataframe(
        df_preds[['team', 'Probability (%)']].head(10), 
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("Explainable AI: What drives success?")
    st.markdown("The model identified **Recent Goal Scoring** and **Recent Wins** as more mathematically important than historical World Cup titles.")
    
    # Create an interactive horizontal bar chart using Plotly
    fig = px.bar(
        df_importance, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Magma'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# 4. Interactive Team Lookup
st.divider()
st.subheader("🔍 Deep Dive: Search for a Team")
team_list = df_preds['team'].sort_values().unique()
selected_team = st.selectbox("Select a national team to view their AI-calculated odds:", team_list)

team_data = df_preds[df_preds['team'] == selected_team].iloc[0]
st.metric(
    label=f"{selected_team}'s Probability to reach Quarter-Finals", 
    value=f"{team_data['Probability (%)']}%"
)