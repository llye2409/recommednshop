import streamlit as st
import pandas as pd

def dataset_overview():
    # --- TITLE ---
    st.title("📦 Dataset Overview")

    # --- 1. CONTEXT ---
    st.header("1. Context")
    st.markdown("""
    This project uses two original datasets:
    - **Products_ThoiTrangNam_raw.csv**: Product information in the Men's Fashion category on Shopee.
    - **Products_ThoiTrangNam_rating_raw.csv**: Product ratings from Shopee users.

    These datasets are the input for the preprocessing process and the development of the recommendation system.
    """)

    # --- 2. SUMMARY TABLE ---
    st.header("2. Summary")
    summary_data = {
        "Dataset": ["Products_ThoiTrangNam_raw.csv", "Products_ThoiTrangNam_rating_raw.csv"],
        "Rows": ["49,653", "1,024,482 "],
        "Columns": [9, 4],
        "Description": [
            "Product listings with metadata",
            "User ratings for each product"
        ]
    }
    st.table(pd.DataFrame(summary_data))

    # --- 3. DETAIL ---
    st.header("3. Detail")

    st.subheader("🛍️ Product Data")
    st.markdown("""
    - **Rows**: 49,653
    - **Columns**: 9
    - **Fields**:
        - `product_id`: Unique product ID.
        - `product_name`: Product name.
        - `category`: Main category (all are "Men's Fashion").
        - `sub_category`: Subcategory (e.g., Shirt, Jeans...).
        - `link`: Link to the product page.
        - `image`: Link to the product image.
        - `price`: Product price (many products have a price of 0).
        - `rating`: Average rating score.
        - `description`: Detailed product description.
    > ⚠️ Raw data may include null values, duplicates, and noise.
    """)

    st.subheader("⭐ Rating Data")
    st.markdown("""
    - **Rows**: 986,805
    - **Columns**: 4
    - **Fields**:
        - `product_id`: Product ID being rated.
        - `user_id`: ID of the user who rated the product.
        - `user`: User's name (many are 'Shopee User').
        - `rating`: Rating score from 1 to 5 (skewed towards 5 stars).
    > ⚠️ Contains duplicated ratings and unclear user identity.
    """)
