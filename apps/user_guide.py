import streamlit as st

# Define the user guide content
def user_guide():
    st.title("📖 User Guide: Recommendation System")
    
    st.markdown("""
    Welcome to the **Recommendation System**! This app provides personalized product recommendations based on your preferences and behavior. Below is a step-by-step guide on how to use the features of this application.
    """)

    # Features Overview
    st.markdown("### 🏠 **Main Features Overview**")
    st.markdown("""
    The application offers several key features that enhance your experience:

    - **Demo Recommended App**: A personalized product recommendation tool that suggests products based on your interactions and search history.
    - **Project Overview**: An overview of the recommendation system, its objectives, and methodologies.
    - **Dataset**: A detailed look at the dataset used in building the recommendation model.
    - **Customer Insights**: Analyzing user behaviors and generating valuable insights for improved recommendations.
    - **Modeling**: A deeper dive into the recommendation algorithms and machine learning models used.
    - **User Guide**: This guide that explains how to use the application.
    - **About Team**: Information about the team behind the recommendation system.
    """)

    # Sidebar Navigation
    st.markdown("### ⚙️ **Navigating the Sidebar**")
    st.markdown("""
    On the left side of the app, you will find a sidebar where you can access all the different pages of the application.

    1. **Main Menu**: Choose from various sections like "Recommended App", "Project Overview", "Dataset", "Customer Insights", etc.
    2. **User Selection**: Select a random user or a product to get tailored recommendations.
    3. **Search Function**: You can search for products by entering search queries. The app will generate recommendations based on your search.
    4. **Random User & Products**: You can refresh the random user and products list by clicking the "🎲 Random" button.
    """)

    # Getting Product Recommendations
    st.markdown("### 🔄 **How to Get Product Recommendations**")
    st.markdown("""
    1. **Login (Optional)**: When you select a user (if logged in), the app will offer personalized recommendations based on that user's previous interactions and preferences.
    2. **Search Recommendations**: 
        - You can enter a search query (e.g., "Simple Squid Game costume") and submit it to get product recommendations based on both your past behavior and the content of the search term. 
        - The app will use **Hybrid Recommendations** that combine content-based filtering and collaborative filtering to give you the best suggestions.
    3. **Viewing Products**: 
        - The app will show a product image and detailed information.
        - If you have selected a product, the app will show **related products** that you might be interested in, based on similar characteristics or user preferences.
    4. **Product Suggestions**: 
        - If you haven't selected any product, the app will show **popular products** based on user ratings and trending items.
    """)

    # Using Features
    st.markdown("### 🛍️ **How to Use the Features**")
    
    st.markdown("#### 1. Search for Products")
    st.markdown("""
    - **Enter a Query**: In the "Search" section, enter a keyword for the product you're interested in.
    - **Submit**: Click "Submit" to see the recommendations related to the search query. The app will display products based on the content of the query and user data.
    """)
    
    st.markdown("#### 2. View Product Details")
    st.markdown("""
    - After selecting a product, the app will show its details along with **similar products** you might like.
    - You can explore these related products in the right column of the layout.
    """)
    
    st.markdown("#### 3. Random User & Products")
    st.markdown("""
    - **Refresh**: Click the "🎲 Random" button in the sidebar to get a random user and product list. This is useful to see how the system can recommend products for different users.
    """)

    # Additional Insights
    st.markdown("### 📊 **Additional Insights**")
    st.markdown("""
    - **Top Products**: If you're not logged in, the app will display top-rated products that are popular among all users.
    - **Customer Insights**: The app can provide you with insights into customer preferences and behaviors that help improve product recommendations.
    """)

    # Version Information
    st.markdown("### 🚀 **Version Information**")
    st.markdown("""
    - **Version**: 1.0
    - **Last Update**: April
    """)    