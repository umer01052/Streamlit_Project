# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
def eda():


    # Streamlit App
    st.title("Exploratory Data Analysis")

    # Load the dataset
    data = pd.read_csv('sampled_dataset.csv')

    # 2. Visualizations
    # Histogram of Daily Vaccinations
    st.write("### Distribution of Daily Vaccinations")
    plt.figure(figsize=(10, 6))
    plt.hist(data['daily_vaccinations'].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Daily Vaccinations')
    plt.xlabel('Daily Vaccinations')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Boxplot of Daily Vaccinations
    st.write("### Boxplot of Daily Vaccinations")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='daily_vaccinations', color='orange')
    plt.title('Boxplot of Daily Vaccinations')
    st.pyplot(plt)

    # Correlation Matrix Heatmap
    st.write("### Correlation Matrix")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # Missing Value Analysis (Bar Plot)
    st.write("### Missing Values by Feature")
    missing_values = data.isnull().sum()
    plt.figure(figsize=(10, 6))
    missing_values[missing_values > 0].sort_values(ascending=False).plot(kind='bar', color='red')
    plt.title('Missing Values by Feature')
    plt.ylabel('Count')
    plt.xlabel('Features')
    st.pyplot(plt)

    # Outlier Detection (Boxplot for Multiple Numeric Columns)
    st.write("### Outlier Detection Across Numeric Features")
    plt.figure(figsize=(12, 8))
    numeric_columns = numeric_data.columns
    sns.boxplot(data=numeric_data, orient='h', palette='Set2')
    plt.title('Outlier Detection Across Numeric Features')
    st.pyplot(plt)

    # Unique Value Counts (Bar Plot)
    st.write("### Unique Value Counts by Feature")
    unique_value_counts = data.nunique()
    plt.figure(figsize=(10, 6))
    unique_value_counts.sort_values(ascending=False).plot(kind='bar', color='purple')
    plt.title('Unique Value Counts by Feature')
    plt.ylabel('Count')
    plt.xlabel('Features')
    st.pyplot(plt)

    # Data Types (Pie Chart)
    st.write("### Data Types Distribution")
    data_types_counts = data.dtypes.value_counts()
    plt.figure(figsize=(8, 8))
    data_types_counts.plot(kind='pie', autopct='%1.1f%%', colors=['gold', 'lightblue', 'pink'], startangle=90)
    plt.title('Data Types Distribution')
    st.pyplot(plt)
