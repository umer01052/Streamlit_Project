import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
def ml():
    # Streamlit App
    st.title("Dynamic Machine Learning Model App")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("#### Dataset Preview:")
        st.write(data.head())

        # Handle Missing Values
        st.write("### Handling Missing Values")
        missing_values = data.isnull().sum()
        st.write("#### Missing Values Before Imputation:")
        st.write(missing_values[missing_values > 0])
        data.fillna(data.median(numeric_only=True), inplace=True)
        st.write("#### Missing Values After Imputation:")
        st.write(data.isnull().sum())

        # Encode Categorical Variables
        st.write("### Encoding Categorical Variables")
        categorical_columns = data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            label_encoders = {}
            for column in categorical_columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                label_encoders[column] = le
            st.write("Categorical variables encoded successfully.")
        else:
            st.write("No categorical variables found.")

        # Scale or Normalize Numerical Features
        st.write("### Scaling Numerical Features")
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        st.write("Numerical features scaled successfully.")

        # Select Target Column
        st.write("### Selecting Target Column")
        target_column = st.selectbox("Select the Target Column", data.columns)

        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.write(f"Training Set: {X_train.shape[0]} samples")
            st.write(f"Testing Set: {X_test.shape[0]} samples")

            # Select Model
            st.write("### Model Selection")
            model_type = st.radio("Choose a model", ("Logistic Regression", "Random Forest"))
            st.write(f"You have selected: **{model_type}**")

            if model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)

                # Evaluate Model
                st.write("### Model Evaluation")
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("#### Classification Report:")
                st.text(classification_report(y_test, y_pred))

                st.write("#### Confusion Matrix:")
                conf_matrix = confusion_matrix(y_test, y_pred)
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                st.pyplot(plt)

            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42, n_estimators=100)
                model.fit(X_train, y_train)

                # Evaluate Model
                st.write("### Model Evaluation")
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("#### Classification Report:")
                st.text(classification_report(y_test, y_pred))

                st.write("#### Confusion Matrix:")
                conf_matrix = confusion_matrix(y_test, y_pred)
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                st.pyplot(plt)

            # Display Processed Data
            st.write("### Processed Data Overview")
            st.write("#### Sample of Processed Training Data:")
            st.write(X_train.head())
            st.write("#### Sample of Processed Testing Data:")
            st.write(X_test.head())
    else:
        st.write("Please upload a dataset to begin.")
