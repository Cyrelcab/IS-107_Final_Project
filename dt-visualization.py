import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import json

#load the .env
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Retail Store Analytics",
    page_icon="📊",
    layout="wide"
)


#function for creating a connection with the database
def db_connection():
    try:
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
        st.error(f"Error connecting to the database: {e}")
        return None


@st.cache_data()
def load_data(query):
    engine = db_connection()
    if engine:
        try:
            with engine.connect() as connection:
                # Execute the query
                data = pd.read_sql_query(text(query), connection)

                # Create date column after loading the data
                data['date'] = pd.to_datetime(
                    dict(
                        year=data['year'],
                        month=data['month'],
                        day=data['day']
                    )
                )

                return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
        finally:
            engine.dispose()  # Dispose of the database connection
    return pd.DataFrame()


query = """
WITH base_data AS (
    SELECT 
        fact_sales."InvoiceNo", 
        fact_sales."CustomerID", 
        dim_pd."Description" AS "Product Name", 
        dim_pd."StockCode", 
        fact_sales."Quantity"::integer AS Quantity, 
        fact_sales."UnitPrice"::float AS "Unit Price", 
        fact_sales.total_amount::float AS "Total Amount", 
        dim_c."Country", 
        dim_t.year, 
        dim_t.month, 
        dim_t.day
    FROM 
        fact_sales
    JOIN dim_product dim_pd 
        ON fact_sales."StockCode" = dim_pd."StockCode"
    JOIN dim_customer dim_c 
        ON dim_c."CustomerID" = fact_sales."CustomerID"
    JOIN dim_time dim_t 
        ON dim_t."date" = fact_sales."date"
)
SELECT 
    "Product Name", 
    "StockCode", 
    Quantity, 
    "Unit Price", 
    "Total Amount", 
    base_data."Country", 
    year, 
    month, 
    day, 
    COUNT(DISTINCT "InvoiceNo")::integer AS total_orders, 
    COUNT(DISTINCT "CustomerID")::integer AS unique_customers
FROM 
    base_data
GROUP BY 
    "Product Name", "StockCode", Quantity, "Unit Price", "Total Amount", base_data."Country", year, month, day
ORDER BY 
    year, month, day;
"""


# Load data
data = load_data(query)

#Sidebar Navigations
with st.sidebar:
    selected = option_menu("Menu", 
                           ["Home", "Overview", "Sales", "Insights", 
                            "Sales Forecasting", "Customer Segmentation"
                            ], 
                           icons=['house', 'speedometer', 'bar-chart', 
                                  'lightbulb', 'graph-up-arrow', 
                                  'people-fill', 'boxes'
                                  ],
                           menu_icon="dashboard", 
                           default_index=0)

#if "Home" is selected in the option_menu 
if selected == 'Home':
    st.title("🏪 Retail Store Analytics Dashboard")
    st.markdown("---")
    st.write("Welcome to the Retail Store Analytics Dashboard! ✨")
    st.write("This dashboard provides insights into sales performance, customer behavior, and product trends Explore various sections to gain a deeper understanding of your retail operations.")

    # Data Overview Section
    st.markdown("## Data Overview")

     # Create tabs for different views of the data
    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Summary Statistics", "ℹ️ Dataset Info"])
    
    with tab1:
        # Display the datasets
        st.markdown("### Online Retail Dataset")
        st.dataframe(
            data.head(10),
            use_container_width=True
        )
        
        # Display total number of records
        st.info(f"Total number of records: {len(data):,}")
        
    with tab2:
        # Display summary statistics
        st.markdown("### Summary Statistics")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Columns**")
            numeric_summary = data.describe()[['quantity', 'Unit Price', 'Total Amount']].round(2)
            st.dataframe(numeric_summary, use_container_width=True)
            
        with col2:
            st.markdown("**Categorical Columns**")
            categorical_summary = pd.DataFrame({
                'Country': data['Country'].value_counts().head(),
                'Products': data['Product Name'].value_counts().head()
            })
            st.dataframe(categorical_summary, use_container_width=True)
    
    with tab3:
        # Display data information
        st.markdown("### Dataset Information")
        
        # Create columns for different metrics
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Total Countries", data['Country'].nunique())
            st.metric("Total Products", data['Product Name'].nunique())
            
        with info_col2:
            st.metric("Date Ranges:", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
            st.metric("Total Orders", data['total_orders'].sum())
            
        with info_col3:
            st.metric("Total Revenue", f"${data['Total Amount'].sum():,.2f}")
            st.metric("Unique Customers", data['unique_customers'].sum())

  # Data Dictionary
    st.markdown("## Data Dictionary")
    
    dict_data = {
        'Column': ['Product Name', 'StockCode', 'quantity', 'Unit Price', 'Total Amount', 'Country', 'year', 'month', 'day', 'total_orders', 'unique_customers'],
        'Description': [
            'Description of the product',
            'Product stock code',
            'Quantity of items purchased',
            'Price per unit',
            'Total price of the transaction',
            'Country where the order was placed',
            'Year of the transaction',
            'Month of the transaction',
            'Day of the transaction',
            'Total number of orders',
            'Number of unique customers'
        ],
        'Type': [
            'String',
            'String',
            'Integer',
            'Float',
            'Float',
            'String',
            'Integer',
            'Integer',
            'Integer',
            'Integer',
            'Integer'
        ]
    }
    
    st.dataframe(
        pd.DataFrame(dict_data),
        use_container_width=True,
        hide_index=True
    )

#if "Overview" is selected in the option_menu 
elif selected == 'Overview':
    st.title("📈 Overview")

    # Date Filter
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = pd.to_datetime(data['date']).min()
        end_date = pd.to_datetime(data['date']).max()
        start_filter = st.date_input("Start Date", start_date, min_value=start_date, max_value=end_date)
    with col_date2:
        end_filter = st.date_input("End Date", end_date, min_value=start_date, max_value=end_date)

    # Filter data
    mask = (data['date'] >= pd.to_datetime(start_filter)) & (data['date'] <= pd.to_datetime(end_filter))
    filtered_data = data[mask]

    # Calculate previous period metrics for comparison
    days_selected = (end_filter - start_filter).days
    previous_start = start_filter - pd.Timedelta(days=days_selected)
    previous_mask = (data['date'] >= pd.to_datetime(previous_start)) & (data['date'] < pd.to_datetime(start_filter))
    previous_data = data[previous_mask]

    # KPIs with Period Comparison
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate current and previous metrics
    total_sales = filtered_data['Total Amount'].sum()
    prev_sales = previous_data['Total Amount'].sum()
    total_orders = filtered_data['total_orders'].sum()
    prev_orders = previous_data['total_orders'].sum()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    prev_avg_order = prev_sales / prev_orders if prev_orders > 0 else 0
    unique_customers = filtered_data['unique_customers'].sum()
    prev_customers = previous_data['unique_customers'].sum()

    # Display KPIs with deltas
    with col1:
        st.metric("Total Revenue", 
                 f"${total_sales:,.2f}", 
                 delta=f"{((total_sales - prev_sales)/prev_sales)*100:.1f}%" if prev_sales > 0 else "N/A")
    with col2:
        st.metric("Total Orders", 
                 f"{total_orders:,}", 
                 delta=f"{((total_orders - prev_orders)/prev_orders)*100:.1f}%" if prev_orders > 0 else "N/A")
    with col3:
        st.metric("Avg Order Value", 
                 f"${avg_order_value:,.2f}", 
                 delta=f"{((avg_order_value - prev_avg_order)/prev_avg_order)*100:.1f}%" if prev_avg_order > 0 else "N/A")
    with col4:
        st.metric("Unique Customers", 
                 f"{unique_customers:,}", 
                 delta=f"{((unique_customers - prev_customers)/prev_customers)*100:.1f}%" if prev_customers > 0 else "N/A")

    # Key Trends
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Revenue Trend
        sales_trend = filtered_data.groupby(pd.Grouper(key='date', freq='M'))['Total Amount'].sum().reset_index()
        trend_fig = px.line(sales_trend, 
                          x='date', 
                          y='Total Amount',
                          title='Monthly Revenue Trend',
                          labels={'totalprice': 'Revenue', 'date': 'Month'})
        st.plotly_chart(trend_fig, use_container_width=True)

    with col_right:
        # Top 5 Products
        top_products = filtered_data.groupby('Product Name')['Total Amount'].sum().nlargest(5).reset_index()
        products_fig = px.bar(top_products,
                            x='Product Name',
                            y='Total Amount',
                            title='Top 5 Products',
                            labels={'totalprice': 'Revenue', 'product_name': 'Product'})
        st.plotly_chart(products_fig, use_container_width=True)

    # Key Insights
    st.markdown("### Key Insights")
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("**Top Market**")
        top_country = filtered_data.groupby('Country')['Total Amount'].sum().nlargest(1)
        st.info(f"🏅 {top_country.index[0]}\n\n${top_country.values[0]:,.2f} in sales")
        
    with insight_col2:
        st.markdown("**Best Seller**")
        top_product = top_products.iloc[0]
        st.info(f"✨ {top_product['Product Name']}\n\n${top_product['Total Amount']:,.2f} in revenue")
        
    with insight_col3:
        st.markdown("**Period Performance**")
        period_growth = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
        st.info(f"📈 {period_growth:,.1f}% growth\n\ncompared to previous period")

#if "Overview" is selected in the option_menu 
elif selected == 'Sales':
    st.title("📊 Sales Performance")
    st.markdown("---")
    
    # Create tabs for different analyses
    sales_tab1, sales_tab2, sales_tab3 = st.tabs(["📈 Time Analysis", "🌍 Geographic Analysis", "📦 Product Analysis"])
    
    with sales_tab1:
        # Date Filter
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = pd.to_datetime(data['date']).min()
            end_date = pd.to_datetime(data['date']).max()
            start_filter = st.date_input("Start Date", start_date, min_value=start_date, max_value=end_date)
        with col_date2:
            end_filter = st.date_input("End Date", end_date, min_value=start_date, max_value=end_date)

        # Filter data based on date range
        mask = (data['date'] >= pd.to_datetime(start_filter)) & (data['date'] <= pd.to_datetime(end_filter))
        filtered_data = data[mask]
        
        # Time Analysis Options
        col_options1, col_options2 = st.columns(2)
        with col_options1:
            time_period = st.selectbox(
                "Select Time Period",
                ["Daily", "Weekly", "Monthly", "Yearly"]
            )
        with col_options2:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line", "Bar", "Area"]
            )

        # Revenue Trend
        if time_period == "Daily":
            sales_over_time = filtered_data.groupby('date')['Total Amount'].sum().reset_index()
            x_axis = 'date'
        elif time_period == "Weekly":
            sales_over_time = filtered_data.groupby(pd.Grouper(key='date', freq='W'))['Total Amount'].sum().reset_index()
            x_axis = 'date'
        elif time_period == "Monthly":
            sales_over_time = filtered_data.groupby(pd.Grouper(key='date', freq='M'))['Total Amount'].sum().reset_index()
            x_axis = 'date'
        else:  # Yearly
            sales_over_time = filtered_data.groupby('year')['Total Amount'].sum().reset_index()
            x_axis = 'year'

        # Create chart based on selection
        if chart_type == "Line":
            trend_fig = px.line(sales_over_time, x=x_axis, y='Total Amount',
                              title=f'Revenue Trends ({time_period})',
                              labels={'totalprice': 'Revenue', 'date': 'Date'})
        elif chart_type == "Bar":
            trend_fig = px.bar(sales_over_time, x=x_axis, y='Total Amount',
                             title=f'Revenue Trends ({time_period})',
                             labels={'totalprice': 'Revenue', 'date': 'Date'})
        else:  # Area
            trend_fig = px.area(sales_over_time, x=x_axis, y='Total Amount',
                              title=f'Revenue Trends ({time_period})',
                              labels={'totalprice': 'Revenue', 'date': 'Date'})
        
        st.plotly_chart(trend_fig, use_container_width=True)

        # Monthly Distribution with Year-over-Year Comparison
        monthly_dist = filtered_data.groupby(['year', 'month'])['Total Amount'].sum().reset_index()
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_dist['month_name'] = monthly_dist['month'].map(month_names)
        monthly_dist['year_month'] = monthly_dist['year'].astype(str) + '-' + monthly_dist['month_name']
        
        # Option to compare years
        show_yoy = st.checkbox("Show Year-over-Year Comparison")
        
        if show_yoy:
            month_sales_fig = px.line(
                monthly_dist,
                x='month_name',
                y='Total Amount',
                color='year',
                title='Monthly Sales Distribution (Year-over-Year)',
                labels={'totalprice': 'Revenue', 'month_name': 'Month', 'year': 'Year'},
                markers=True
            )
            # Customize layout for better readability
            month_sales_fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue ($)",
                legend_title="Year"
            )
        else:
            # Original monthly view (aggregated across years)
            monthly_agg = filtered_data.groupby('month')['Total Amount'].sum().reset_index()
            monthly_agg['month_name'] = monthly_agg['month'].map(month_names)
            monthly_agg = monthly_agg.sort_values('month')
            
            month_sales_fig = px.line(
                monthly_agg,
                x='month_name',
                y='Total Amount',
                title='Monthly Sales Distribution (All Years)',
                labels={'Total Amount': 'Revenue', 'month_name': 'Month'},
                markers=True
            )
            
        st.plotly_chart(month_sales_fig, use_container_width=True)

    with sales_tab2:
        # Geographic Analysis
        col_geo1, col_geo2 = st.columns(2)
        
        with col_geo1:
            # Sales by Country Map
            sales_by_country = filtered_data.groupby('Country')['Total Amount'].sum().reset_index()
            country_fig = px.choropleth(
                sales_by_country,
                locations='Country',
                locationmode='country names',
                color='Total Amount',
                title='Sales Distribution by Country',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(country_fig, use_container_width=True)
            
        with col_geo2:
            # Country Metrics
            metric_option = st.selectbox(
                "Select Metric",
                ["Total Revenue", "Average Order Value", "Total Orders", "Unique Customers"]
            )
            
            country_metrics = filtered_data.groupby('Country').agg({
                'Total Amount': 'sum',
                'total_orders': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
            
            country_metrics['avg_order_value'] = country_metrics['Total Amount'] / country_metrics['total_orders']
            
            if metric_option == "Total Revenue":
                y_col = 'Total Amount'
                title = 'Total Revenue by Country'
            elif metric_option == "Average Order Value":
                y_col = 'avg_order_value'
                title = 'Average Order Value by Country'
            elif metric_option == "Total Orders":
                y_col = 'total_orders'
                title = 'Total Orders by Country'
            else:
                y_col = 'unique_customers'
                title = 'Unique Customers by Country'
            
            country_metric_fig = px.bar(
                country_metrics.sort_values(y_col, ascending=True).tail(10),
                x=y_col,
                y='Country',
                orientation='h',
                title=title
            )
            st.plotly_chart(country_metric_fig, use_container_width=True)

    with sales_tab3:
        # Product Analysis
        col_prod1, col_prod2 = st.columns(2)
        
        with col_prod1:
            # Top Products Configuration
            top_n = st.slider('Number of Top Products', 5, 20, 10)
            sort_by = st.selectbox(
                "Sort Products By",
                ["Revenue", "Quantity", "Orders"]
            )
            
            if sort_by == "Revenue":
                sort_col = 'Total Amount'
                title = f'Top {top_n} Products by Revenue'
            elif sort_by == "Quantity":
                sort_col = 'quantity'
                title = f'Top {top_n} Products by Quantity Sold'
            else:
                sort_col = 'total_orders'
                title = f'Top {top_n} Products by Number of Orders'
            
            top_products = filtered_data.groupby('Product Name')[sort_col].sum().nlargest(top_n).reset_index()
            
            product_fig = px.bar(
                top_products,
                x='Product Name',
                y=sort_col,
                title=title
            )
            product_fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(product_fig, use_container_width=True)
            
        with col_prod2:
            # Product Performance Metrics
            st.markdown("### Product Performance Metrics")
            product_metrics = filtered_data.groupby('Product Name').agg({
                'Total Amount': 'sum',
                'quantity': 'sum',
                'total_orders': 'sum'
            }).reset_index()
            
            product_metrics['avg_price_per_unit'] = product_metrics['Total Amount'] / product_metrics['quantity']
            
            # Add metric formatting
            product_metrics['Total Amount'] = product_metrics['Total Amount'].apply(lambda x: f"${x:,.2f}")
            product_metrics['avg_price_per_unit'] = product_metrics['avg_price_per_unit'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(
                product_metrics.sort_values('Total Amount', ascending=False).head(top_n),
                use_container_width=True
            )

    # Summary Metrics
    st.markdown("### Summary Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        total_revenue = filtered_data['Total Amount'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")

    with metric_col2:
        total_orders = filtered_data['total_orders'].sum()
        st.metric("Total Orders", f"{total_orders:,}")

    with metric_col3:
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        st.metric("Average Order Value", f"${avg_order_value:,.2f}")

    with metric_col4:
        unique_customers = filtered_data['unique_customers'].sum()
        st.metric("Unique Customers", f"{unique_customers:,}")

    # Interactive Data Table
    st.markdown("### Detailed Sales Data")
    show_data = st.checkbox("Show Raw Data")
    if show_data:
        # Add search functionality
        search_term = st.text_input("Search products:")
        
        display_data = filtered_data[['date', 'Product Name', 'quantity', 'Unit Price', 'Total Amount', 'Country']]
        
        if search_term:
            display_data = display_data[display_data['Product Name'].str.contains(search_term, case=False)]
        
        st.dataframe(
            display_data.sort_values('date', ascending=False),
            use_container_width=True
        )

#if "Insights" is selected in the option_menu 
elif selected == "Insights":
    st.title("📑 Insights")
    st.markdown("---")
    
    # Calculate key metrics for insights
    total_revenue = data['Total Amount'].sum()
    total_orders = data['total_orders'].sum()
    unique_customers = data['unique_customers'].sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    # High-level insights in cards
    st.markdown("### 🎯 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with kpi_col2:
        st.metric("Total Orders", f"{total_orders:,}")
    
    with kpi_col3:
        st.metric("Unique Customers", f"{unique_customers:,}")
        
    with kpi_col4:
        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    
    # Detailed insights
    st.markdown("### 🔍 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Top market analysis
        top_country = data.groupby('Country')['Total Amount'].sum().nlargest(1)
        st.markdown("**🌍 Top Market**")
        st.info(
            f"**{top_country.index[0]}**\n\n"
            f"Revenue: ${top_country.values[0]:,.2f}"
        )
    
    with insight_col2:
        # Best selling product
        top_product = data.groupby('Product Name')['Total Amount'].sum().nlargest(1)
        st.markdown("**🏆 Best Selling Product**")
        st.info(
            f"**{top_product.index[0]}**\n\n"
            f"Revenue: ${top_product.values[0]:,.2f}"
        )
    
    with insight_col3:
        # Customer engagement
        avg_customer_value = total_revenue / unique_customers if unique_customers > 0 else 0
        st.markdown("**👥 Customer Engagement**")
        st.info(
            f"**Average Customer Value**\n\n"
            f"${avg_customer_value:,.2f} per customer"
        )
    
    # Trend Analysis
    st.markdown("### 📈 Trend Analysis")
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Monthly sales trend
        monthly_sales = data.groupby(['year', 'month'])['Total Amount'].sum().reset_index()
        monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
        
        sales_trend_fig = px.line(
            monthly_sales,
            x='date',
            y='Total Amount',
            title='Monthly Sales Trend'
        )
        sales_trend_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales ($)",
            showlegend=False
        )
        st.plotly_chart(sales_trend_fig, use_container_width=True)
    
    with trend_col2:
        # Top 5 countries
        top_countries = data.groupby('Country')['Total Amount'].sum().nlargest(5).reset_index()
        
        country_fig = px.bar(
            top_countries,
            x='Country',
            y='Total Amount',
            title='Top 5 Countries by Revenue'
        )
        country_fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Revenue ($)",
            showlegend=False
        )
        st.plotly_chart(country_fig, use_container_width=True)
    
    # Additional insights
    st.markdown("### 💡 Additional Insights")
    
    # Calculate and display product diversity
    total_products = data['Product Name'].nunique()
    avg_products_per_order = data['quantity'].mean()
    
    add_col1, add_col2 = st.columns(2)
    
    with add_col1:
        st.info(
            f"**Product Diversity**\n\n"
            f"• Total unique products: {total_products:,}\n"
            f"• Average items per order: {avg_products_per_order:.1f}"
        )
    
    with add_col2:
        # Calculate customer geographic distribution
        customer_countries = data['Country'].nunique()
        top_3_countries = data.groupby('Country')['Total Amount'].sum().nlargest(3)
        
        st.info(
            f"**Geographic Reach**\n\n"
            f"• Active in {customer_countries} countries\n"
            f"• Top 3 markets: {', '.join(top_3_countries.index)}"
        )

#if "Sales Forecasting" is selected in the option_menu 
elif selected == 'Sales Forecasting':
    st.title("📊 Sales Forecasting")
    st.markdown("---")
    
    # Create tabs for different time granularities
    tab1, tab2, tab3 = st.tabs(["Daily Forecast", "Weekly Forecast", "Monthly Forecast"])
    
    # Load JSON data
    with open('daily.json') as f:
        daily_data = json.load(f)
    with open('weekly.json') as f:
        weekly_data = json.load(f)
    with open('monthly.json') as f:
        monthly_data = json.load(f)
    
    # Daily Forecast Tab
    with tab1:
        st.subheader("Daily Sales Forecast")
        
        # Convert historical data
        daily_dates = list(daily_data['historical_sales'].keys())
        daily_values = list(daily_data['historical_sales'].values())
        
        # Create forecast dates (next 12 days)
        last_date = pd.to_datetime(daily_dates[-1])
        forecast_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                         for i in range(len(daily_data['forecasted_sales']))]
        
        # Create plot
        fig_daily = go.Figure()
        
        # Add historical data
        fig_daily.add_trace(go.Scatter(
            x=daily_dates,
            y=daily_values,
            name='Historical Sales',
            line=dict(color='blue')
        ))
        
        # Add forecasted data
        fig_daily.add_trace(go.Scatter(
            x=forecast_dates,
            y=daily_data['forecasted_sales'],
            name='Forecasted Sales',
            line=dict(color='red', dash='dash')
        ))
        
        fig_daily.update_layout(
            title='Daily Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)

        add_col1, add_col2, add_col3 = st.columns(3)

        with add_col1:
            st.info(f"**Total Predicted Sales**\n\n${daily_data['total_predicted_sales']:,.2f}")

        with add_col2:
            st.info(f"**Total Actual Sales**\n\n${daily_data['total_actual_sales']:,.2f}")

        with add_col3:
            st.info(f"**Prediction Difference**\n\n${daily_data['prediction_difference']:,.2f}")
    
    # Weekly Forecast Tab
    with tab2:
        st.subheader("Weekly Sales Forecast")
        
        # Convert historical data
        weekly_dates = list(weekly_data['historical_sales'].keys())
        weekly_values = list(weekly_data['historical_sales'].values())
        
        # Create forecast dates (next 12 weeks)
        last_date = pd.to_datetime(weekly_dates[-1])
        forecast_dates = [(last_date + pd.Timedelta(weeks=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                         for i in range(len(weekly_data['forecasted_sales']))]
        
        # Create plot
        fig_weekly = go.Figure()
        
        # Add historical data
        fig_weekly.add_trace(go.Scatter(
            x=weekly_dates,
            y=weekly_values,
            name='Historical Sales',
            line=dict(color='blue')
        ))
        
        # Add forecasted data
        fig_weekly.add_trace(go.Scatter(
            x=forecast_dates,
            y=weekly_data['forecasted_sales'],
            name='Forecasted Sales',
            line=dict(color='red', dash='dash')
        ))
        
        fig_weekly.update_layout(
            title='Weekly Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
        st.metric("Total Predicted Sales", f"${weekly_data['total_predicted_sales']:,.2f}")
        st.metric("Total Actual Sales", f"${weekly_data['total_actual_sales']:,.2f}")
        st.metric("Prediction Difference", f"${weekly_data['prediction_difference']:,.2f}")
    
    # Monthly Forecast Tab
    with tab3:
        st.subheader("Monthly Sales Forecast")
        
        # Convert historical data
        monthly_dates = list(monthly_data['historical_sales'].keys())
        monthly_values = list(monthly_data['historical_sales'].values())
        
        # Create forecast dates (next 12 months)
        last_date = pd.to_datetime(monthly_dates[-1])
        forecast_dates = []
        for i in range(len(monthly_data['forecasted_sales'])):
            # Use pd.DateOffset for monthly increments
            next_date = last_date + pd.DateOffset(months=i+1)
            forecast_dates.append(next_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Create plot
        fig_monthly = go.Figure()
        
        # Add historical data
        fig_monthly.add_trace(go.Scatter(
            x=monthly_dates,
            y=monthly_values,
            name='Historical Sales',
            line=dict(color='blue')
        ))
        
        # Add forecasted data
        fig_monthly.add_trace(go.Scatter(
            x=forecast_dates,
            y=monthly_data['forecasted_sales'],
            name='Forecasted Sales',
            line=dict(color='red', dash='dash')
        ))
        
        fig_monthly.update_layout(
            title='Monthly Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.metric("Total Predicted Sales", f"${monthly_data['total_predicted_sales']:,.2f}")
        st.metric("Total Actual Sales", f"${monthly_data['total_actual_sales']:,.2f}")
        st.metric("Prediction Difference", f"${monthly_data['prediction_difference']:,.2f}")

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit by HM (Hadlok Malabwan)")