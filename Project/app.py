# Importing necessary libraries and modules
import streamlit as st
from introduction import introduction
from dataset_overview import dataset_overview
from EDAa import eda
from dp import data_preprocessing
from ml import ml
from conclusion_and_insights import conclusion_and_insights

# Configure the Streamlit page
st.set_page_config(
    page_title="Interactive Project Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Title
st.title("COVID-19 Vaccination Project Dashboard")
st.write("Welcome to the COVID-19 Vaccination Project dashboard for exploring the project, performing EDA, building models, and deriving insights.")

# Define sections and map them to functions
sections = {
    "Introduction": introduction,
    "DataSet Overview": dataset_overview,
    "EDA (Exploratory Data Analysis)": eda,
    "Data Preprocessing": data_preprocessing,
    "Model": ml,
    "Conclusion and Insights": conclusion_and_insights,
}

# Sidebar for navigation
st.sidebar.title("COVID-19 Vaccination Project")

selected_section = st.sidebar.radio("Go to", list(sections.keys()))

# Display the selected section
st.markdown("---")  # Add a horizontal line for better UI separation
sections[selected_section]()