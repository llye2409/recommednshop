import streamlit as st
import pandas as pd
import pickle
import json
import base64
import os
from sklearn.metrics.pairwise import cosine_similarity
import re
from underthesea import pos_tag, word_tokenize

@st.cache_resource
def load_vectorizer():
    with open('models/vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_tfidf_matrix():
    with open('models/tfidf_matrix.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_svd_model():
    with open("models/svd_model_balanced.pkl", "rb") as f:
        return pickle.load(f)

# Hàm đọc file CSS
@st.cache_resource
def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# hàm đọc file csv tạo thành datafram
@st.cache_data
def load_csv_file(file_path, encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
    except pd.errors.ParserError:
        print(f"Lỗi khi đọc file CSV '{file_path}'.")
    return None

# hàm đọc file json
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
    except json.JSONDecodeError:
        print(f"File '{file_path}' không đúng định dạng JSON.")
    return None

# Load stop words
STOP_WORD_FILE = 'data/vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# Load irrelevant words
with open('data/irrelevant.txt', 'r', encoding='utf-8') as file:
    irrelevant_word = file.read()

irrelevant_words = irrelevant_word.split('\n')

# Load ảnh local dạng base64 để nhúng vào HTML
@st.cache_data
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

    
# Hàm cắt bớt text
def truncate_text(text, max_length=100):
    text = str(text) if text else ""
    return text[:max_length] + "..." if len(text) > max_length else text

# Hàm khởi tạo session
def init_session_state():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "show_login_form" not in st.session_state:
        st.session_state.show_login_form = False


def render_header():
    with st.container():
        

        # Dòng 1: Tên ứng dụng
        st.markdown("<h2>🛒 Recommendation Shopee</h2>", unsafe_allow_html=True)

        # Dòng 2: Mô tả ngắn
        st.markdown(
            """
            <p style='font-size:16px; color:#555;'>
                Ứng dụng gợi ý sản phẩm sử dụng Collaborative Filtering  & Content-Based Filtering
            </p>
            """,
            unsafe_allow_html=True
        )

        # Dòng 3: Hướng dẫn sử dụng
        with st.expander("📘 Hướng dẫn sử dụng"):
            st.markdown(
                """
                **Các tính năng chính:**
                - 🔍 **Chọn User** để xem các sản phẩm được gợi ý riêng cho người dùng đó.
                - 🛒 **Chọn sản phẩm** để xem chi tiết và các sản phẩm liên quan.
                - 🔐 **Tìm kiếm** sản phẩm bằng từ khoá (VD: 'giày', 'áo thun',...).

                **Cách sử dụng:**
                1. Chọn một **User** và/hoặc một **Sản phẩm** ở phía trên.
                2. Kéo xuống để xem **danh sách gợi ý** hoặc **chi tiết sản phẩm**.
                3. Dùng thanh tìm kiếm để khám phá các sản phẩm theo từ khoá mong muốn.
                """
            )

        
def clean_text(input_text):
    """Làm sạch văn bản theo các bước tiền xử lý"""

    global stop_words
    global irrelevant_words
    
    if not input_text:  # Check if input is empty
        return ""

    # Step 1: Convert text to lowercase
    text_lower = input_text.lower()
    
    # Step 2: Delete link (URL)
    text_lower = re.sub(r"https?://\S+|www\.\S+", "", text_lower)
    
    # Step 3: Remove special characters, numbers and punctuation (keep only letters and spaces)
    text_no_special = re.sub(r"[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]", "", text_lower)
    
    # Step 4: Tokenize and remove stopwords
    tokens_vn = word_tokenize(text_no_special, format="text").split()
    filtered_tokens_vn = [token for token in tokens_vn if token not in stop_words]
    text_no_stopwords_vn = " ".join(filtered_tokens_vn)
    
    # Step 5: Label the word types, keeping only nouns (N), verbs (V) and adjectives (A)
    pos_result = pos_tag(text_no_stopwords_vn)
    important_tokens = [word for word, pos in pos_result if pos.startswith("N") or pos.startswith("V") or pos.startswith("A")]
    
    # Step 6: Filter out irrelevant words
    final_tokens = [word for word in important_tokens if word not in irrelevant_words]
    
    return " ".join(final_tokens)

def recommend_products_by_search(query, products_df, vectorizer, tfidf_matrix, top_n=6):
    query_cleaned = clean_text(query)  # 🔹 Làm sạch văn bản tìm kiếm
    query_vector = vectorizer.transform([query_cleaned])  # 🔹 Vector hóa sau khi làm sạch

    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(products_df)][:top_n]

    if not top_indices:
        return pd.DataFrame()

    results_df = products_df.iloc[top_indices].copy()
    results_df["similarity"] = cosine_sim[top_indices]

    return results_df


def render_search_result(search_query, products_df, placeholder_image, vectorizer, tfidf_matrix):
    # search_query = st.session_state.get("search_query", "").strip()

    if search_query:
        # Gợi ý sản phẩm theo nội dung mô tả/tên sản phẩm
        suggested_df = recommend_products_by_search(
            search_query, products_df, vectorizer, tfidf_matrix, top_n=6
        )

        st.markdown("## 🔎 Kết quả tìm kiếm")

        if suggested_df.empty:
            st.warning("❌ Không tìm thấy sản phẩm phù hợp.")
        else:
            cols = st.columns(3)
            for i, (_, row) in enumerate(suggested_df.iterrows()):
                with cols[i % 3]:
                    render_product_card_product(row, placeholder_image)

        st.stop()



# Hàm tạo form đăng nhập
def render_login_form(ratings_df):
    if not st.session_state.user_id and st.session_state.show_login_form:
        with st.form("login_form"):
            st.markdown("## Đăng nhập")
            user_input = st.text_input("Nhập user_id của bạn", placeholder="VD: 148")
            submitted = st.form_submit_button("Đăng nhập")
            if submitted:
                if user_input.isdigit():
                    user_id = int(user_input)
                    if user_id in ratings_df["user_id"].values:
                        st.session_state.user_id = user_id
                        st.success(f"🎉 Đăng nhập thành công! Xin chào User {user_id}")
                        st.rerun()
                    else:
                        st.error("❌ User ID không tồn tại trong hệ thống.")
                else:
                    st.error("⚠️ Vui lòng nhập user_id là số.")


def get_recommended_products(user_id, users_df, products_df):
    try:
        user_id = int(user_id)
        user_row = users_df[users_df["user_id"] == user_id]
        if not user_row.empty:
            top_products = user_row.iloc[0]["top_products"]
            return products_df[products_df["product_id"].isin(top_products)]
    except:
        st.warning("user_id không hợp lệ.")
    return pd.DataFrame()


def render_product_card_product(row, placeholder_image):
    """Hiển thị thẻ sản phẩm (product card) từ dòng dữ liệu (row)"""
    image_path = row.get('image', '')
    if not image_path or str(image_path).strip().lower() in ['nan', 'none', 'null', '']:
        image_path = placeholder_image

    # Tạo HTML cho ảnh sản phẩm
    if os.path.exists(image_path):
        base64_img = image_to_base64(image_path)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image">'
    elif str(image_path).startswith("http"):
        image_html = f'<img src="{image_path}" alt="Product Image">'
    else:
        base64_img = image_to_base64(placeholder_image)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image">'

    # Lấy đánh giá và tạo chuỗi sao
    rating = row.get('rating', 0)
    try:
        rating = float(rating)
    except:
        rating = 0
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    star_html = '⭐' * full_stars + ('✨' if half_star else '')

    # Tạo thẻ HTML hiển thị
    st.markdown(f"""
    <div class="card-wrapper">
        <div class="card">
            <div class="card-img">{image_html}</div>
            <div class="card-title">
                <a href="/?product_id={row['product_id']}" style="text-decoration: none; color: inherit;">
                    {truncate_text(row.get('product_name', 'Không tên'), max_length=60)}
                </a>
            </div>
            <div class="card-id"><strong>Mã sản phẩm:</strong> {row.get('product_id', 'N/A')}</div>
            <div class="card-subcat"><strong>Danh mục:</strong> {row.get('sub_category', 'Không rõ')}</div>
            <div class="card-rating">Đánh giá: {star_html} ({rating:.1f}/5)</div>
            <div class="card-price">Giá: {int(row.get('price', 0)):,}đ</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị mô tả trong expander
    with st.expander("📄 Xem mô tả chi tiết sản phẩm"):
        st.markdown(row.get('description', 'Không có mô tả'))



# Hiển thị moule sản phẩm liên quan
def render_recommended_products_ralated(related_products, placeholder_image):
    st.markdown("<h3 style='margin-top:40px;'>🧩 Sản phẩm liên quan</h3>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (_, row) in enumerate(related_products.iterrows()):
        with cols[i % 3]:
            render_product_card_product(row, placeholder_image)

# Hiển thị moule sản phẩm gợi ý cho user
def render_recommended_products_for_user(recommendation_products, placeholder_image):
    # st.markdown("<h3 style='margin-top:40px;'>🔁 Có thể bạn thích</h3>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (_, row) in enumerate(recommendation_products.iterrows()):
        with cols[i % 3]:
            render_product_card_product(row, placeholder_image)





# Hàm hiển thi chi tiết sản phẩm
def render_detail_product(product, placeholder_image):
    # Ảnh
    image_path = product.get('image', '')
    if not image_path or str(image_path).strip().lower() in ['nan', 'none', 'null', '']:
        image_path = placeholder_image

    if os.path.exists(image_path):
        base64_img = image_to_base64(image_path)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" />'
    elif str(image_path).startswith("http"):
        image_html = f'<img src="{image_path}" />'
    else:
        base64_img = image_to_base64(placeholder_image)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" />'

    # HTML layout
    st.markdown("[🔙 Quay về trang chính](/)", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="product-detail-container">
        <div class="product-detail-left">
            {image_html}
        </div>
        <div class="product-detail-right">
            <div class="product-name">{product.get('product_name', 'Không tên')}</div>
            <div class="product-rating">⭐⭐⭐⭐☆</div>
            <div class="product-price">{int(product.get('price', 0)):,}đ</div>
            <div class="product-category">Danh mục: {product.get('category', 'Không rõ')}</div>
            <button class="buy-button" onclick="alert('Chức năng mua hàng chưa kích hoạt')">🛍 Mua ngay</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mô tả sản phẩm
        # Mô tả sản phẩm (ẩn trong expander)
    with st.expander("📝 Xem mô tả sản phẩm"):
        st.markdown(f"""
        <div class="product-description-content">
            {product.get('description', 'Không có mô tả.')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    

def get_similar_products(product_id, products_df, tfidf_matrix, top_n=6):
    if product_id not in products_df['product_id'].values:
        return pd.DataFrame()

    idx_list = products_df.index[products_df['product_id'] == product_id].tolist()
    if not idx_list:
        return pd.DataFrame()

    idx = idx_list[0]

    if idx >= tfidf_matrix.shape[0]:
        return pd.DataFrame()

    product_vec = tfidf_matrix[idx]

    # Tính cosine giữa sản phẩm được chọn và toàn bộ sản phẩm
    cosine_scores = cosine_similarity(product_vec, tfidf_matrix).flatten()

    # Sắp xếp và lấy top chỉ số sản phẩm tương tự
    similar_indices = cosine_scores.argsort()[::-1]

    # Loại bỏ chính nó
    similar_indices = [i for i in similar_indices if i != idx]

    # Lọc chỉ số hợp lệ để tránh lỗi
    valid_indices = [i for i in similar_indices if i < len(products_df)]

    return products_df.iloc[valid_indices[:top_n]]


def get_recommendations_for_user(user_id, ratings_df, model, top_n=10):
    if user_id not in ratings_df['user_id'].unique():
        return pd.DataFrame([], columns=['product_id', 'EstimateScore'])

    all_products = ratings_df['product_id'].unique()
    rated_products = ratings_df[ratings_df['user_id'] == user_id]['product_id'].unique()
    products_to_predict = [pid for pid in all_products if pid not in rated_products]

    predictions = []
    for pid in products_to_predict:
        try:
            pred = model.predict(user_id, pid).est
            predictions.append((pid, pred))
        except:
            continue  # Bỏ qua nếu model không thể dự đoán

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_products = pd.DataFrame(predictions[:top_n], columns=['product_id', 'EstimateScore'])
    return top_products
