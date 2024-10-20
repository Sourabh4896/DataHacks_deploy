# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient
from textblob import TextBlob

# Initialize the Gemini API client
client = InferenceClient(api_key="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")

# Function to send feedback to the Gemini API
def send_feedback_to_gemini(feedback_text):
    try:
        response = ""
        for message in client.chat_completion(
            model="mistralai/Mistral-Nemo-Instruct-2407",
            messages=[{"role": "user", "content": feedback_text}],
            max_tokens=500,
            stream=True,
        ):
            response += message.choices[0].delta.content
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Set the title of the app
st.title("Customer Feedback Analysis Dashboard")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Page", ("Home", "Upload Data", "Sales Analysis", "Restaurant Information", "EDA", "Performance Metrics", "Sentiment Analysis", "Actionable Insights", "Send Feedback"))

# Home Page
if options == "Home":
    st.subheader("Welcome to the Customer Feedback Analysis Dashboard")
    st.write("This application helps you analyze customer feedback data through exploratory data analysis, performance metrics, and sentiment analysis.")

# Data Upload Page
elif options == "Upload Data":
    st.title("Upload Data")
    st.write("Please upload the CSV files containing feedback data.")
    
    # Upload CSV files
    food_feedback_file = st.file_uploader("Upload Food Feedback CSV", type=["csv"])
    service_feedback_file = st.file_uploader("Upload Service Feedback CSV", type=["csv"])
    
    if st.button("Process Data"):
        if food_feedback_file and service_feedback_file:
            st.success("Files uploaded successfully!")
            st.session_state['food_feedback'] = pd.read_csv(food_feedback_file)
            st.session_state['service_feedback'] = pd.read_csv(service_feedback_file)
        else:
            st.error("Please upload both CSV files.")

# Sales Analysis Page
elif options == "Sales Analysis":
    st.title("Sales Data Analysis")
    sales_file = st.file_uploader("Upload Sales Data CSV", type=["csv"])
    
    if st.button("Process Sales Data"):
        if sales_file:
            sales_data = pd.read_csv(sales_file)
            st.session_state['sales_data'] = sales_data
            st.success("Sales data uploaded successfully!")
            
            # Display sales data
            st.write("### Sales Data Preview")
            st.dataframe(sales_data.head())
            
            # Summary statistics
            st.write("### Summary Statistics")
            st.write(sales_data.describe())
            
            # Visualizations
            st.write("### Total Sales Over Time")
            sales_data['Date'] = pd.to_datetime(sales_data['Date'])
            total_sales = sales_data.groupby('Date')['Order_Total'].sum().reset_index()
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=total_sales, x='Date', y='Order_Total')
            plt.title('Total Sales Over Time')
            plt.xlabel('Date')
            plt.ylabel('Total Sales')
            st.pyplot(plt)
            
            # Sales by Payment Type
            st.write("### Sales by Payment Type")
            payment_type_sales = sales_data.groupby('Payment_Type')['Order_Total'].sum().reset_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(data=payment_type_sales, x='Payment_Type', y='Order_Total', palette='Set2')
            plt.title('Sales by Payment Type')
            plt.xlabel('Payment Type')
            plt.ylabel('Total Sales')
            st.pyplot(plt)
            
            # Sales by Order Type
            st.write("### Sales by Order Type")
            order_type_sales = sales_data.groupby('Order_Type')['Order_Total'].sum().reset_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(data=order_type_sales, x='Order_Type', y='Order_Total', palette='Set2')
            plt.title('Sales by Order Type')
            plt.xlabel('Order Type')
            plt.ylabel('Total Sales')
            st.pyplot(plt)
            
        else:
            st.error("Please upload the sales data CSV.")

# Restaurant Information Analysis Page
elif options == "Restaurant Information":
    st.title("Restaurant Information Analysis")
    restaurant_file = st.file_uploader("Upload Restaurant Information CSV", type=["csv"])
    
    if st.button("Process Restaurant Data"):
        if restaurant_file:
            restaurant_data = pd.read_csv(restaurant_file)
            st.session_state['restaurant_data'] = restaurant_data
            st.success("Restaurant data uploaded successfully!")
            
            # Display restaurant data
            st.write("### Restaurant Data Preview")
            st.dataframe(restaurant_data.head())
            
            # Display summary statistics
            st.write("### Summary Statistics")
            st.write(restaurant_data.describe(include='all'))
            
            # Visualization of restaurant types
            if 'Restaurant_Type' in restaurant_data.columns:
                st.write("### Distribution of Restaurant Types")
                restaurant_type_counts = restaurant_data['Restaurant_Type'].value_counts()
                plt.figure(figsize=(12, 6))
                sns.barplot(x=restaurant_type_counts.index, y=restaurant_type_counts.values, palette='Set2')
                plt.title('Distribution of Restaurant Types')
                plt.xlabel('Restaurant Type')
                plt.ylabel('Count')
                st.pyplot(plt)
            else:
                st.warning("No 'Restaurant_Type' column found in the data.")

            # Top 5 Restaurants by Rating
            if 'Rating' in restaurant_data.columns:
                st.write("### Top 5 Restaurants by Rating")
                top_restaurants = restaurant_data.nlargest(5, 'Rating')
                st.dataframe(top_restaurants[['Restaurant_Name', 'Rating']])

            # Value Counts for Location
            if 'Location' in restaurant_data.columns:
                st.write("### Restaurant Counts by Location")
                location_counts = restaurant_data['Location'].value_counts()
                st.bar_chart(location_counts)

        else:
            st.error("Please upload the restaurant information CSV.")

# EDA Page
elif options == "EDA":
    if 'food_feedback' in st.session_state and 'service_feedback' in st.session_state:
        food_feedback = st.session_state['food_feedback']
        service_feedback = st.session_state['service_feedback']
        
        st.subheader("Exploratory Data Analysis")
        
        # Visualizations
        st.write("### Food Feedback Distribution")
        plt.figure(figsize=(10, 5))
        sns.countplot(data=food_feedback, x='Answer_ff', palette='Set2')
        st.pyplot(plt)

        st.write("### Service Feedback Distribution")
        plt.figure(figsize=(10, 5))
        sns.countplot(data=service_feedback, x='Answer_sf', palette='Set2')
        st.pyplot(plt)

    else:
        st.error("Please upload data first.")

# Performance Metrics Page
elif options == "Performance Metrics":
    if 'food_feedback' in st.session_state and 'service_feedback' in st.session_state:
        food_feedback = st.session_state['food_feedback']
        service_feedback = st.session_state['service_feedback']
        
        st.subheader("Customer Satisfaction Scores")
        
        # Calculate CSAT for food and service feedback
        def calculate_csat(feedback_column):
            positive_responses = feedback_column.str.contains('good|excellent|great|satisfied|happy', case=False, na=False).sum()
            total_responses = len(feedback_column)
            return (positive_responses / total_responses) * 100

        csat_food = calculate_csat(food_feedback['Answer_ff'])
        csat_service = calculate_csat(service_feedback['Answer_sf'])
        
        st.write(f"**Customer Satisfaction Score for Food:** {csat_food}%")
        st.write(f"**Customer Satisfaction Score for Service:** {csat_service}%")
    else:
        st.error("Please upload data first.")

# Sentiment Analysis Page
elif options == "Sentiment Analysis":
    if 'food_feedback' in st.session_state and 'service_feedback' in st.session_state:
        food_feedback = st.session_state['food_feedback']
        service_feedback = st.session_state['service_feedback']

        # Clean text and perform sentiment analysis
        def get_sentiment(text):
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        
        food_feedback['Sentiment_Score'] = food_feedback['Answer_ff'].apply(get_sentiment)
        service_feedback['Sentiment_Score'] = service_feedback['Answer_sf'].apply(get_sentiment)

        # Classify sentiment
        food_feedback['Sentiment_Type'] = food_feedback['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
        service_feedback['Sentiment_Type'] = service_feedback['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

        st.subheader("Sentiment Distribution for Food Feedback")
        food_sentiment_counts = food_feedback['Sentiment_Type'].value_counts()
        st.bar_chart(food_sentiment_counts)

        st.subheader("Sentiment Distribution for Service Feedback")
        service_sentiment_counts = service_feedback['Sentiment_Type'].value_counts()
        st.bar_chart(service_sentiment_counts)

    else:
        st.error("Please upload data first.")

# Actionable Insights Page
elif options == "Actionable Insights":
    if 'food_feedback' in st.session_state and 'service_feedback' in st.session_state:
        food_feedback = st.session_state['food_feedback']
        service_feedback = st.session_state['service_feedback']

        st.subheader("Actionable Insights")
        # Logic for actionable insights
        st.write("Generating insights based on feedback...")
        # This is just a placeholder
        st.write("1. Improve menu options based on food feedback.")
        st.write("2. Enhance service training based on service feedback.")
    else:
        st.error("Please upload data first.")

# Send Feedback Page
elif options == "Send Feedback":
    st.title("Send Feedback")
    
    feedback_text = st.text_area("Please enter your feedback:")
    
    if st.button("Submit Feedback"):
        if feedback_text:
            response = send_feedback_to_gemini(feedback_text)
            st.success("Feedback sent to Gemini API!")
            st.write("Response from Gemini API:")
            st.write(response)
        else:
            st.error("Please enter feedback before submitting.")
# Sales Data Analysis Page
elif options == "Sales Data Analysis":
    st.title("Sales Data Analysis")
    sales_file = st.file_uploader("Upload Sales Data CSV", type=["csv"])
    
    if st.button("Analyze Sales Data"):
        if sales_file:
            sales_data = pd.read_csv(sales_file)
            st.write("### Sales Data Preview")
            st.dataframe(sales_data.head())  # Display the first few rows of the data
            
            # Print column names for debugging
            st.write("### Columns in the dataset")
            st.write(sales_data.columns.tolist())  # Display the list of column names
            
            try:
                # Grouping by date and calculating total sales
                total_sales = sales_data.groupby('Date')['Order_Total'].sum().reset_index()
                st.write("### Total Sales Over Time")
                st.line_chart(total_sales.set_index('Date'))
            except KeyError as e:
                st.error(f"Column not found: {e}")
                st.write("Please make sure that the 'Date' and 'Order_Total' columns are present in the dataset.")
        else:
            st.error("Please upload the sales data CSV.")
