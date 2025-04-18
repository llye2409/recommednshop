import streamlit as st
import pandas as pd
import json

#Read data from JSON
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
customers_insights = load_json_data('data/customer_insights.json')

def generate_insights():
    st.title("📊 Customer Insights")
    st.markdown("""
        <div style="background-color:#F2F0C9; padding:16px; border-radius:10px">
            <span style="font-size:18px">🧹 <strong>Note:</strong> All charts and analysis results below are based on <strong>cleaned data</strong>.</span>
            <ul>
                <li>Duplicate products or missing key information have been removed.</li>
                <li>Invalid ratings (out of 1-5 range or missing info) have been filtered out.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 1. System Overview
    st.subheader("1️⃣ Summary")
    #total_products = len(products_df)
    total_products = customers_insights['Summary']['Products']
    unique_users = customers_insights['Summary']['Users']
    total_ratings = customers_insights['Summary']['Ratings']
    products_with_ratings = customers_insights['Summary']['Rated_Products']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Products", f"{total_products:,}")
    col2.metric("👥 Users", f"{unique_users:,}")
    col3.metric("⭐ Ratings", f"{total_ratings:,}")
    col4.metric("📊 Rated Products", f"{products_with_ratings:,} ({products_with_ratings/total_products:.0%})")

    st.info("✅ Nearly **100% of products** have at least one rating, indicating good review coverage.")

    # 2. Most Rated Products
    most_rated_products = customers_insights['Most_Rated_Products']  # Giả sử đây là key trong file JSON
    df_most_rated = pd.DataFrame(most_rated_products)
    st.subheader("2️⃣ Most Rated Products")
    st.dataframe(df_most_rated, use_container_width=True)
    st.image('assets/images/most_rated_products.png', use_container_width =True)

    # 3. Top Rated Products (with ≥50 reviews)
    st.subheader("3️⃣ Top Rated Products (≥50 Reviews)")
    high_rating = customers_insights['Top_Rated_Products']
    st.dataframe(high_rating, use_container_width=True)
    st.image('assets/images/top_rated_products.png', use_container_width =True)

    # 4. Review Distribution by Users
    st.subheader("4️⃣ Distribution of Reviews per User")
    st.image('assets/images/user_review_distribution.png', use_container_width =True)


    st.info("📌 Most users review only 1–3 products. Very few users leave many reviews.")

    # 5. Rating Score Distribution
    st.subheader("5️⃣ Rating Score Distribution")
    st.image('assets/images/rating_distribution.png', use_container_width =True)

    st.info("🎯 The distribution leans toward **high ratings (4–5 stars)**, indicating positive user experiences.")

    # 6. Products by Category
    st.image('assets/images/products_by_category.png', use_container_width =True)
    st.dataframe(customers_insights['Products_by_Category'], use_container_width=True)

    # 7. Analysis by Price Level#
    st.subheader("7️⃣ Analysis by Price Range")
    st.dataframe(customers_insights['Analysis_by_Price_Range'], use_container_width=True)
    st.image('assets/images/price_range.png', use_container_width =True)

    # 8. WordCloud of Product Descriptions
    st.subheader("8️⃣ WordCloud of Product Descriptions")
    st.image('assets/images/WordCloud-of-Product.png', use_container_width =True)


    st.info("☁️ The WordCloud highlights frequently used keywords in product descriptions.")

    st.markdown("---")
    st.subheader("✅ Recommended Strategies from Data")

    st.success("""
    - 🎯 **Target active users**: They're loyal and responsive to offers.
    - ⭐ **Recommend top-rated products**: More likely to be trusted and purchased.
    - 🎁 **Encourage reviews on new/low-review products** with vouchers, mini-games.
    - 🧠 **Recommend based on popular categories and price levels** to align with most user behavior.
    """)
