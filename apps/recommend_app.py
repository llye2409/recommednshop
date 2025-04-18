import streamlit as st
from lib.mylib import *

def sidebar_controls(products_df):
    with st.sidebar:
        st.title("📋 Gợi ý sản phẩm")
        st.caption("📝 Chọn user hoặc sản phẩm hoặc tất cả để nhận gợi ý phù hợp.")

        # Users
        st.markdown("### 👤 USERS")
        user_options = ["-- Chưa đăng nhập --"] + list(map(str, st.session_state.get("random_users", [])))
        selected_user = st.selectbox("Chọn người dùng", user_options)
        user_id = int(selected_user) if selected_user != "-- Chưa đăng nhập --" else None

        # Products
        st.markdown("### 🏷️ PRODUCTS")
        product_list = [("-- Chọn sản phẩm --", None)]
        random_products_df = st.session_state.get("random_products_df")
        if random_products_df is not None:
            product_list += [(row["product_name"], row["product_id"]) for _, row in random_products_df.iterrows()]

        selected_product = st.selectbox("Chọn sản phẩm", product_list, format_func=lambda x: x[0])
        selected_product_id = selected_product[1]

        # Search form
        st.markdown("### 🔍 SEARCH")
        with st.form(key="search_form"):
            query = st.text_area("Ví dụ: Bộ đồ hóa trang nhân vật phim Squid Game thiết kế đơn giản", 
                                 height=120, max_chars=2000, placeholder="Nhập nội dung tìm kiếm").strip()
            submit = st.form_submit_button("Submit")

        return {
            "user_id": user_id,
            "selected_product_id": selected_product_id,
            "query": query,
            "submit": submit
        }

# Recommend app
def recommend_app(products_df, recommendations_df, user_ids, vectorizer, tfidf_matrix):
    render_header()

    # ========== SIDEBAR ==========
    if "random_products_df" not in st.session_state:
        st.session_state.random_products_df = get_random_products(products_df)
    if "random_users" not in st.session_state:
        st.session_state.random_users = get_random_users(user_ids)

    with st.sidebar:
        st.caption("🌀 Random user & Products khác (tùy chọn)")
        if st.button("🎲 Random"):
            st.session_state.random_users = get_random_users(user_ids)
            st.session_state.random_products_df = get_random_products(products_df)

    filters = sidebar_controls(products_df)
    user_id = filters["user_id"]
    selected_product_id = filters["selected_product_id"]
    search_query = filters["query"]
    submitted = filters["submit"]
    placeholder_image = 'assets/images/No_Image_Available.jpg'

    # ========== MAIN CONTENT ==========
    if user_id:
        user_name = recommendations_df.loc[recommendations_df['user_id'] == user_id, 'user'].iloc[0]
        st.success(f"Xin chào {user_name}!")

        if submitted and search_query:
            st.markdown("## 🔄 Gợi ý theo tìm kiếm (Hybrid)")
            st.caption("📝 Gợi ý sản phẩm dựa trên nội dung và hành vi người dùng")
            st.markdown(
                f"""
                🔍 Kết quả tìm kiếm cho <span style='color: #EE4D2D; font-weight: bold;'>'{truncate_text(search_query, max_length = 200)}'</span>
                """,
                unsafe_allow_html=True
            )
            results = recommend_personalized_by_search(search_query, products_df, vectorizer, tfidf_matrix, recommendations_df)
            if not results.empty:
                render_scrollable_products_html(results, placeholder_image)
            else:
                st.warning("Không tìm thấy sản phẩm phù hợp.")
            return

        if selected_product_id:
            left_col, right_col = st.columns([1, 2])
            with left_col:
                st.markdown("### 🛍️ Sản phẩm đang xem")
                show_selected_product(products_df, selected_product_id, placeholder_image, tfidf_matrix)

            with right_col:
                st.markdown("### 🔄 Sản phẩm tương tự (Hybrid)")
                st.caption("📝 Gợi ý sản phẩm tương tự với sản phẩm đang xem và hành vi người dùng")
                
                results = recommend_personalized_related_products(selected_product_id, products_df, tfidf_matrix, recommendations_df)
                render_scrollable_products_html (results, placeholder_image)

        else:
            st.markdown("## 🔄 Gợi ý sản phẩm (Collaborative Filtering)")
            st.caption("📝 Gợi ý các sản phẩm yêu thích chung chung dựa trên cá nhân hóa")
            user_recs = recommendations_df[recommendations_df['user_id'] == user_id].merge(products_df, on='product_id', how='left').dropna().head(6)
            render_scrollable_products_html(user_recs, placeholder_image)
    else:
        if submitted and search_query:
            st.markdown("## 🔄 Gợi ý sản phẩm (Content-Based Filtering)")
            st.caption("📝 Gợi ý các sản phẩm dựa trên nội dung")
            st.markdown(
                f"""
                🔍 Kết quả tìm kiếm cho <span style='color: #EE4D2D; font-weight: bold;'>'{truncate_text(search_query, max_length = 200)}'</span>
                """,
                unsafe_allow_html=True
            )
            results = recommend_products_by_search(search_query, products_df, vectorizer, tfidf_matrix)
            if not results.empty:
                render_scrollable_products_html(results, placeholder_image)
                # thêm sort result nếu có
            else:
                st.warning("Không tìm thấy sản phẩm phù hợp.")
        

        if selected_product_id:
            left_col, right_col = st.columns([1, 2])
            with left_col:
                st.markdown("### 🛍️ Sản phẩm đang xem")
                show_selected_product(products_df, selected_product_id, placeholder_image, tfidf_matrix)
                results = recommend_related_products(selected_product_id, products_df, tfidf_matrix)
            with right_col:
                st.markdown("## 🔄 Gợi ý sản phẩm (Content-Based Filtering)")
                st.caption("📝 Gợi ý sản phẩm dựa trên nội dung tương tự với sản phẩm đang xem")
                render_scrollable_products_html(results, placeholder_image)
        else:
            st.markdown("## 🔥 Sản phẩm nổi bật")
            st.caption("📝 Vì bạn chưa **Đăng nhập**, shop gợi ý trước cho bạn một số sản phẩm phổ biến nhé!")
            top = products_df.sort_values("rating", ascending=False).head(6)
            render_scrollable_products_html(top, placeholder_image)
