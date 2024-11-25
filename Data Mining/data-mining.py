from dotenv import load_dotenv
import pandas as pd
import json
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os

# Function to connect to PostgreSQL Database using SQLAlchemy
def connect_to_db():
    try:
        #Load the environment variables
        load_dotenv()

        # Get credentials from environment variables
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASS')

        # Create a database connection
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
        
        return engine
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Function to save the analysis results as JSON
def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Customer Segmentation Analysis (Clustering)
def customer_segmentation(df, n_clusters=3):
    # Assuming df has 'Annual_Spend' and 'Age' columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Annual_Spend', 'Age']])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    
    # Save customer segmentation results to JSON
    segmentation_results = df[['CustomerID', 'Cluster']].to_dict(orient='records')
    save_to_json(segmentation_results, 'customer_segmentation.json')
    print("Customer segmentation saved to 'customer_segmentation.json'")

# Sales Forecasting Analysis (ARIMA Model)
def sales_forecasting(df, column='Sales', period=12, freq='M'):
    """
    Forecast future sales for given time period and frequency (monthly, weekly, daily).
    
    :param df: DataFrame containing sales data.
    :param column: Column name for sales data.
    :param period: Forecast period (number of time steps ahead).
    :param freq: Frequency for forecasting ('M' for monthly, 'W' for weekly, 'D' for daily).
    
    :return: None, saves forecasted data to JSON file.
    """
    # Assuming df has 'Date' and 'Sales' columns
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resampling based on frequency
    if freq == 'M':
        df_resampled = df.resample('M').sum()  # Resample to monthly data
    elif freq == 'W':
        df_resampled = df.resample('W').sum()  # Resample to weekly data
    elif freq == 'D':
        df_resampled = df.resample('D').sum()  # Resample to daily data
    else:
        raise ValueError("Invalid frequency. Use 'M' for monthly, 'W' for weekly, 'D' for daily.")
    
    # Fit ARIMA model
    model = ARIMA(df_resampled[column], order=(5, 1, 0))  # p, d, q
    model_fit = model.fit()

    # Forecast the next 'period' data points (e.g., 12 months)
    forecast = model_fit.forecast(steps=period)
    
    # Save forecast results to JSON
    forecast_results = {
        'forecast': forecast.tolist(),
        'dates': pd.date_range(df_resampled.index[-1], periods=period + 1, freq=freq).strftime('%Y-%m-%d').tolist()
    }
    
    save_to_json(forecast_results, f'sales_forecasting_{freq}.json')
    print(f"Sales forecasting (frequency: {freq}) saved to 'sales_forecasting_{freq}.json'")

# Product Forecasting Analysis (ARIMA Model)
def product_forecasting(df, product_column='ProductSales', period=12, freq='M'):
    """
    Forecast future product sales for given time period and frequency (monthly, weekly, daily).
    
    :param df: DataFrame containing product sales data.
    :param product_column: Column name for product sales data.
    :param period: Forecast period (number of time steps ahead).
    :param freq: Frequency for forecasting ('M' for monthly, 'W' for weekly, 'D' for daily).
    
    :return: None, saves forecasted data to JSON file.
    """
    # Assuming df has 'Date' and product sales columns
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resampling based on frequency
    if freq == 'M':
        df_resampled = df.resample('M').sum()  # Resample to monthly data
    elif freq == 'W':
        df_resampled = df.resample('W').sum()  # Resample to weekly data
    elif freq == 'D':
        df_resampled = df.resample('D').sum()  # Resample to daily data
    else:
        raise ValueError("Invalid frequency. Use 'M' for monthly, 'W' for weekly, 'D' for daily.")
    
    # Fit ARIMA model
    model = ARIMA(df_resampled[product_column], order=(5, 1, 0))  # p, d, q
    model_fit = model.fit()

    # Forecast the next 'period' data points (e.g., 12 months)
    forecast = model_fit.forecast(steps=period)
    
    # Save forecast results to JSON
    forecast_results = {
        'forecast': forecast.tolist(),
        'dates': pd.date_range(df_resampled.index[-1], periods=period + 1, freq=freq).strftime('%Y-%m-%d').tolist()
    }
    
    save_to_json(forecast_results, f'product_forecasting_{freq}.json')
    print(f"Product forecasting (frequency: {freq}) saved to 'product_forecasting_{freq}.json'")

# Main function to load data, perform analysis, and save results
def main():
    engine = connect_to_db()
    if engine is None:
        print("Failed to connect to the database.")
        return
    
    # Load data from database (example: query a table into a DataFrame)
    query = "SELECT * FROM your_table;"  # Replace with your actual query
    df = pd.read_sql(query, engine)
    
    # Perform Customer Segmentation
    customer_segmentation(df)
    
    # Perform Sales Forecasting
    sales_forecasting(df, freq='M')  # Monthly forecast
    sales_forecasting(df, freq='W')  # Weekly forecast
    sales_forecasting(df, freq='D')  # Daily forecast
    
    # Perform Product Forecasting
    product_forecasting(df, product_column='ProductSales', freq='M')  # Monthly forecast
    product_forecasting(df, product_column='ProductSales', freq='W')  # Weekly forecast
    product_forecasting(df, product_column='ProductSales', freq='D')  # Daily forecast
    
    print("Analysis completed and saved to respective JSON files.")

# Run the main function
if __name__ == "__main__":
    main()
