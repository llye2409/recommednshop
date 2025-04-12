import streamlit as st
import pandas as pd
from lib.mylib import * 

# Cấu hình trang
st.set_page_config(
    page_title="🎯 Hệ thống gợi ý sản phẩm",
    layout="centered",
    page_icon="📊"
)

# Tải CSS
load_css("css/style/card.css")
load_css("css/style/detail.css")
load_css("css/style/header.css")

# Tải dữ liệu
products_df = load_csv_file("data/products_df.csv")
# ratings_df = load_csv_file("data/ratings.csv")
recommendations_df = load_csv_file("data/recommendations_by_user.csv")
user_ids = recommendations_df['user_id'].unique()

# Tải mô hình và vectorizer
vectorizer = load_vectorizer()
tfidf_matrix = load_tfidf_matrix()
# model_content_svd = load_svd_model()

# Ảnh mặc định khi không có ảnh sản phẩm
placeholder_image = 'images/No_Image_Available.jpg'

def render_selected_product(product_id):
    """Hiển thị chi tiết sản phẩm khi chọn sản phẩm."""
    product = products_df[products_df['product_id'].astype(str) == str(product_id)].squeeze()
    if product.empty:
        st.error("Không tìm thấy sản phẩm.")
    else:
        st.markdown("## 🛒 Chi tiết sản phẩm")
        render_detail_product(product, placeholder_image)

        st.markdown("### 🔁 Sản phẩm liên quan:")
        similar_products = get_similar_products(int(product_id), products_df, tfidf_matrix)
        if not similar_products.empty:
            render_recommended_products_ralated(similar_products, placeholder_image)

def show_selected_product(product_id):
    """Hiển thị chi tiết sản phẩm khi chọn sản phẩm."""
    product = products_df[products_df['product_id'].astype(str) == str(product_id)].squeeze()
    if product.empty:
        st.error("Không tìm thấy sản phẩm.")
    else:
        st.markdown("## 🛒 Chi tiết sản phẩm")
        render_detail_product(product, placeholder_image)


def main():
    render_header()

    # Khởi tạo
    recommend_products = None
    section_title = None
    selected_user = None
    selected_product_id = None
    selected_product_name = None
    search_query = ""

    # Kiểm tra URL có product_id không
    query_params = st.query_params
    product_id_from_url = query_params.get("product_id", None)
    if product_id_from_url:
        render_selected_product(product_id_from_url)
        st.stop()

    # === Chọn phương thức gợi ý ===
    st.markdown("## 🧠 Chọn phương thức gợi ý")
    option = st.radio(
        "🔍 Bạn muốn hệ thống gợi ý theo cách nào?",
        [
            "🔗 Gợi ý theo người dùng (Collaborative Filtering)",
            "📦 Gợi ý theo sản phẩm (Content-Based Filtering)",
            "🔍 Tìm kiếm sản phẩm (Content-Based Filtering)"
        ]
    )

    # === Gợi ý theo người dùng ===
    if option.startswith("🔗"):
        selected_user = st.selectbox("👤 Chọn người dùng:", user_ids)
        if selected_user:
            section_title = f"🎯 Gợi ý cho người dùng {selected_user}"
            user_recs = recommendations_df[recommendations_df['user_id'] == selected_user]
            recommend_products = user_recs.merge(products_df, on='product_id', how='left')
            recommend_products = recommend_products.dropna(subset=["product_name", "price"])

    # === Gợi ý theo sản phẩm (chọn từ danh sách) ===
    elif option.startswith("📦"):
        product_list = [(row["product_name"], row["product_id"]) for _, row in products_df.iterrows()]
        selected_product_name, selected_product_id = st.selectbox(
            "📦 Chọn sản phẩm từ danh sách:", product_list, format_func=lambda x: x[0]
        )

        if selected_product_id:
            # 👉 Hiển thị chi tiết sản phẩm trước
            show_selected_product(selected_product_id)

            # 👉 Sau đó mới gợi ý các sản phẩm liên quan
            section_title = f"🔁 Sản phẩm liên quan đến: {selected_product_name}"
            recommend_products = get_similar_products(int(selected_product_id), products_df, tfidf_matrix)

    # === Tìm kiếm sản phẩm (nhập từ khóa) ===
    elif option.startswith("🔍"):
        with st.form("search_form_1"):  # Để tránh trùng key
            search_query = st.text_input("🔍 Nhập từ khóa sản phẩm:")
            search_button = st.form_submit_button("Tìm kiếm sản phẩm")

        if search_button and search_query:
            recommend_products = recommend_products_by_search(
                search_query, products_df, vectorizer, tfidf_matrix
            )
            section_title = f"🔍 Kết quả tìm kiếm cho: '{search_query}'"

    # === Hiển thị danh sách gợi ý nếu có ===
    if recommend_products is not None and not recommend_products.empty:
        st.markdown(f"## {section_title}")
        render_recommended_products_for_user(recommend_products, placeholder_image)
    else:
        if selected_user or selected_product_id or (search_button and search_query):
            st.info("Không tìm thấy sản phẩm phù hợp.")

if __name__ == "__main__":
    main()
