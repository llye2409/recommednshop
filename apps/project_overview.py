import streamlit as st

def Project_Overview():

    st.title("🎯 Product Recommendation System for Shopee.vn")

    st.markdown("""
    This project aims to build a **Recommendation System** for [Shopee.vn](https://shopee.vn) – one of the leading e-commerce platforms in Vietnam.  
    The main goal is to **personalize the user experience**, suggesting products based on product content and user interaction behavior.
    """)

    st.info("""
    📝 **Note**  
    This project is part of the **Graduation Thesis** at the **IT Center – University of Science, HCMC**.  
    The goal is to apply knowledge in **Data Analysis and Machine Learning** to solve a **real-world problem in e-commerce**, specifically product recommendation systems.
    """)

    st.subheader("💡 Project Scope")
    st.markdown("""
    Focused on the **Men's Fashion** category, using product data and user ratings to develop two recommendation models:

    - **Content-based Filtering**: Suggest similar products based on product names, descriptions, and content features.
    - **Collaborative Filtering**: Suggest products based on the behaviors and preferences of other users with similar tastes.
    """)

    st.subheader("📦 Dataset")
    st.markdown("""
    - `products.csv`: Detailed information about nearly **49,000** products after cleaning.
    - `ratings.csv`: Over **840,000** user ratings.
    """)
    st.info("""
    📝 **Note**  
    The data provided is for educational purposes only and should not be shared or used for any other purposes.
    """)

    st.subheader("🛠️ Technologies & Libraries")
    st.markdown("""
    - **Language**: Python  
    - **Data Processing**: Pandas, PySpark  
    - **Modeling**: scikit-learn, Gensim, SurPRISE  
    - **Vietnamese NLP**: underthesea  
    - **Main Algorithms**: TF-IDF, Cosine Similarity, Surprise SVD
    """)

    st.subheader("🔄 Project Workflow")
    st.image('assets/images/work-flow-datata-science.png', use_container_width=True)

    st.subheader("📈 Expected Outcomes")
    st.markdown("""
    The system will help **Shopee** improve **user retention** and enhance **conversion rates**  
    through **accurate**, **smart**, and **personalized** product recommendations.
    """)
