import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_overview():
    st.title("Dataset Overview")

    # Load dataset
    uploaded_file = "sampled_dataset.csv"  # Use the uploaded file
    df = pd.read_csv(uploaded_file)

    # Dataset Information
    st.markdown("### **Dataset Information**")
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")
    st.markdown("**Column Names and Data Types:**")
    st.write(df.dtypes)

    # Visual: Column Data Types Distribution
    st.markdown("### **Data Types Distribution**")
    data_types = df.dtypes.value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=data_types.index.astype(str), y=data_types.values, ax=ax)
    ax.set_title("Column Data Types Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Preview Dataset
    st.markdown("### **Dataset Preview**")
    st.dataframe(df.head())

    # Summary Statistics
    st.markdown("### **Summary Statistics**")
    st.write(df.describe())

    # Missing Data
    st.markdown("### **Missing Data**")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.write(missing_data[missing_data > 0])

        # Visual: Missing Data Heatmap
        st.markdown("### **Missing Data Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No missing data found!")

    # Visual: Numerical Columns Distributions
    st.markdown("### **Sample Distributions**")
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    for column in numerical_columns:
        st.markdown(f"#### {column}")
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

# Run the dataset overview function
if __name__ == "__main__":
    dataset_overview()
