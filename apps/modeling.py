import streamlit as st
import pandas as pd

def model_page():
    st.title("🤖 Model Overview")
    st.subheader("🧠 Machine Learning Algorithms Used")

    with st.expander("Content-Based Filtering"):
        st.subheader("Cosine Similarity")
        st.markdown("""
        - Compares similarity between products by measuring the angle between two feature vectors.
        - Vectors are built from product descriptions, names, and attributes.
        """)

        st.subheader("🧾 TF-IDF with Gensim")
        st.markdown("""
        - Text modeling using word frequency and importance.
        - Utilized **Gensim** library for processing and similarity calculation with **SparseMatrixSimilarity**.
        """)

    with st.expander("👥 Collaborative Filtering"):
        st.subheader("🛠️ Surprise Library")
        st.markdown("""
        - Easy to use, supports multiple recommendation algorithms like **SVD**.
        - Input data includes user-product rating matrix.
        """)

    st.subheader("⚖️ Why These Methods & Their Advantages")

    data = {
        "Method": ["Content-Based", "Collaborative"],
        "Main Advantages": [
            "Does not require user data\nTransparent recommendations\nNo product cold-start issue",
            "Surprising suggestions\nCaptures latent preferences"
        ],
        "Reason for Selection": [
            "Suitable when interaction data is limited",
            "Effective with abundant user ratings"
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.subheader("📊 Model Evaluation")
    st.markdown("""
    - **Collaborative Filtering (SVD)**: RMSE = `0.8478`
    - **Content-Based Filtering**: Precision / Recall *(not measured yet)*
    """)

    st.subheader("🔧 Future Improvements")
    st.markdown("""
    - 🎛️ Hyperparameter tuning (TF-IDF, SVD, etc.)
    - 🔗 Hybrid model combining multiple methods
    - 📍 Context-aware personalization (location, time, etc.)
    - 😄 Sentiment analysis from product reviews
    - ⚖️ Addressing data imbalance
    - 📈 Benchmarking to choose optimal models
    """)

    st.subheader("🚀 Real-World Deployment")
    st.markdown("""
    **System Integration**:
    - 🧾 Content-Based: Recommendations based on currently viewed product.
    - 👥 Collaborative: Recommendations based on user behavior.
    """)

    st.subheader("📊 Prediction Visualization")
    st.image("assets/images/top_recommendations_user5000.png", caption="Top 10 recommended products for user 5000", use_container_width=True)
