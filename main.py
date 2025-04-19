import streamlit as st
from apps.recommend_app import recommend_app
from apps.project_overview import Project_Overview
from apps.dataset import dataset_overview
from apps.customer_insight import generate_insights
from apps.about_team import about_team
from apps.modeling import model_page
from apps.user_guide import user_guide
from lib.mylib import *

# Set page config
st.set_page_config(
    page_title="Recommendation System",
    page_icon="🤖",
    layout="wide", 
    initial_sidebar_state="expanded"
)

def load_resources():
    load_css("assets/css/style/card.css")
    load_css("assets/css/style/detail.css")
    load_css("assets/css/style/header.css")

    products_df = load_csv_file("data/products_df.csv")
    ratings_df = load_csv_file("data/ratings.csv")
    recommendations_df = load_csv_file("data/recommendations_by_user.csv")
    user_ids = list(recommendations_df['user_id'].unique())
    vectorizer = load_vectorizer()
    tfidf_matrix = load_tfidf_matrix()

    return products_df, ratings_df, recommendations_df, user_ids, vectorizer, tfidf_matrix

def main():
    with st.spinner("🔄 Vui lòng chờ một chút nhé..."):
        global products_df, ratings_df, recommendations_df, user_ids, vectorizer, tfidf_matrix
        products_df, ratings_df, recommendations_df, user_ids, vectorizer, tfidf_matrix = load_resources()

    # ========== SIDEBAR ==========
    with st.sidebar:
        selected = st.selectbox(
            "📂 Menu",
            [
                "🏠 Demo Recommended App",
                "📖 Project Overview",
                "📊 Dataset",
                "🧠 Customer Insights",
                "⚙️ Modeling",
                "📘 User Guide",
                "👨‍💻 About team"
            ]
        )
        
    # ========== MAIN CONTENT ==========
    if selected == "🏠 Demo Recommended App":
        recommend_app(products_df, ratings_df, recommendations_df, user_ids, vectorizer, tfidf_matrix)
    elif selected == "📖 Project Overview":
        Project_Overview()
    elif selected == "📊 Dataset":
        dataset_overview()
    elif selected == "🧠 Customer Insights":
        generate_insights()
    elif selected == "⚙️ Modeling":
        model_page()
    elif selected == "📘 User Guide":
        user_guide()
    elif selected == "👨‍💻 About team":
        about_team()

    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Footer information
        st.markdown("""<div style='font-size: 16px'>
            <b>🎓 BÁO CÁO TỐT NGHIỆP: DS - ML:</b><br>
            <b>📌 Chủ đề:</b> Recommendation System<br>
            <b>🏫 Lớp:</b> DL07_k302<br>
            <b>📅 Ngày báo cáo:</b> 19/04/2025<br>
            <b>👩‍🏫 GVHD:</b> Khuất Thùy Phương<br>          
            <b>👥 Team Members:</b><br>
            - Nguyễn Văn Duy<br>
            - Phạm Vũ An<br>
            - Nguyễn Văn Gulist<br>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 13px; color: gray; text-align:center'>
            📌 <i>Version: 1.0 | Update: 04/2025</i>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()