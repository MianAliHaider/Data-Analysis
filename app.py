import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model and encoders
model = joblib.load('model.pkl')
brand_channel_encoder = joblib.load('Brand_channel_encoder.pkl')
primary_language_encoder = joblib.load('Primary_language_encoder.pkl')
category_encoder = joblib.load('Category_encoder.pkl')
country_encoder = joblib.load('Country_encoder.pkl')

# Function to display Introduction
def introduction():
    st.title('Data Science Project: Predicting Youtube Subscribers 2024 (millions)')
    st.markdown("""
    ## Introduction
    This project aims to predict the number of subscribers (in millions) for different channels on a platform. The dataset includes information such as the brand channel, primary language, category, country, and the number of subscribers. 
    The goal is to build a predictive model using a Random Forest Regressor to estimate the number of subscribers.
    """)

# Plot the Distribution of Subscribers
def plot_subscribers_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df['Subscribers (millions)'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Subscribers (millions)')
    ax.set_xlabel('Subscribers (millions)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Function to display EDA
def eda(df):
    st.title('Exploratory Data Analysis')

    # Plot Distribution of Subscribers
    plot_subscribers_distribution(df)
    
    # Select only numeric columns for correlation
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_columns]

    # Compute and display the correlation matrix
    correlation_matrix = df_numeric.corr()
    st.subheader('Correlation Heatmap')
    st.write(correlation_matrix)

    # Visualize the correlation heatmap
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)  # Pass the figure explicitly

# Function to prepare data and split into training and testing sets
def prepare_data():
    # Load your dataset
    df = pd.read_csv('youtube_subscribers_data.csv')
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    df['Brand channel'] = label_encoder.fit_transform(df['Brand channel'])
    df['Primary language'] = label_encoder.fit_transform(df['Primary language'])
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Country'] = label_encoder.fit_transform(df['Country'])
    
    # Split into features and target
    X = df.drop(columns=['Subscribers (millions)', 'Name'])  # Drop 'Name' if it's just an identifier
    y = df['Subscribers (millions)']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, df

# Function to display Model Results
def model_section(X_test, y_test):
    st.title('Model Results')
    st.markdown("""
    ## Random Forest Regressor Model
    The Random Forest Regressor model was trained to predict the number of subscribers (millions) based on features such as brand channel, primary language, category, and country.
    """)

    # Get predictions and evaluation metrics
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('Model Evaluation')
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    # Show sample predictions
    predictions_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred})
    st.subheader('True vs Predicted Values')
    st.write(predictions_df.head())

# Function to display Conclusion
def conclusion():
    st.title('Conclusion')
    st.markdown("""
    ## Conclusion
    In this project, we built a predictive model using Random Forest Regressor to predict the number of subscribers for a platform. The model performed well, as seen from the R-squared value. 
    We performed thorough exploratory data analysis (EDA) to understand the distribution of features and their correlation with the target variable, 'Subscribers (millions)'. 

    The next steps would involve further model tuning and potential deployment for real-time predictions.
    """)

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    options = ['Introduction', 'EDA', 'Model', 'Conclusion']
    choice = st.sidebar.radio("Select Section", options)

    # Load dataset
    df = pd.read_csv('youtube_subscribers_data.csv')

    # Handle different sections based on user choice
    if choice == 'Introduction':
        introduction()
    elif choice == 'EDA':
        eda(df)
    elif choice == 'Model':
        # Prepare data and split it
        X_train, X_test, y_train, y_test, df = prepare_data()
        model_section(X_test, y_test)  # Pass the test data to model section
    elif choice == 'Conclusion':
        conclusion()

# Run the Streamlit app
if __name__ == '__main__':
    main()
