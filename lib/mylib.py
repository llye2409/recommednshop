import streamlit as st
import pandas as pd
import pickle
import json
import base64
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import re
from underthesea import pos_tag, word_tokenize
import streamlit.components.v1 as components

# Load vectorizer model (Cached resource)
@st.cache_resource
def load_vectorizer():
    with open('models/vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

# Load TF-IDF matrix (Cached resource)
@st.cache_resource
def load_tfidf_matrix():
    with open('models/tfidf_matrix.pkl', 'rb') as f:
        return pickle.load(f)

# Load SVD model (Cached resource)
@st.cache_resource
def load_svd_model():
    with open("models/svd_model_balanced.pkl", "rb") as f:
        return pickle.load(f)

# Function to read and apply CSS (Cached resource)
@st.cache_resource
def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Function to read CSV file and create DataFrame (Cached data)
@st.cache_data
def load_csv_file(file_path, encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except pd.errors.ParserError:
        print(f"Error reading CSV file '{file_path}'.")
    return None

# Function to read JSON file
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except json.JSONDecodeError:
        print(f"File '{file_path}' is not valid JSON format.")
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

# Load local image as base64 for embedding in HTML (Cached data)
@st.cache_data
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Function to truncate text to a specified max length
def truncate_text(text, max_length=100):
    text = str(text) if text else ""
    return text[:max_length] + "..." if len(text) > max_length else text

# Function to initialize session state
def init_session_state():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "show_login_form" not in st.session_state:
        st.session_state.show_login_form = False

# Function to get random users from user IDs
def get_random_users(user_ids, n=5):
    return random.sample(user_ids, min(n, len(user_ids)))

# Function to get random products from the product DataFrame
def get_random_products(products_df, n=5):
    return products_df.sample(n=n)

# Function to render the header section
def render_header():
    with st.container():
        st.image("assets/images/shopee-banner.png", use_container_width =True)
        st.divider()

        
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

def keyword_search(query, products_df):
    # clean query
    query = query.lower().strip()
    keywords = query.split()

    # make mask filter
    mask = products_df['product_name'].str.lower()
    for kw in keywords:
        mask = mask[mask.str.contains(kw)]

    results = products_df.loc[mask.index].copy()

    if results.empty:
        return pd.DataFrame()
        
    return results


def filter_products_by_category(products_df, query):
    query = query.strip().lower() 
    categories = products_df["sub_category"].dropna().str.lower().unique()  # Get list of categories

    # Check if the keyword matches any category
    if query in categories:
        # If yes, filter the DataFrame by that category
        filtered_df = products_df[products_df["sub_category"].str.lower() == query]
        return filtered_df
    else:
        # If the length is too short (less than 3 words), switch to simple keyword search
        if len(query.split()) <= 3:
            filtered_df = keyword_search(query, products_df)
            return filtered_df
    
    # If no match, return the unmodified DataFrame
    return products_df

def recommend_products_by_search(query, products_df, vectorizer, tfidf_matrix, top_n=10):
    
    # If the length is too short (less than 3 words), switch to simple keyword search
    products_df = filter_products_by_category(products_df, query)
    query_cleaned = clean_text(query)  # 🔹 Clean the search text
    query_vector = vectorizer.transform([query_cleaned])  # 🔹 Vectorize after cleaning

    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(products_df)][:top_n]

    if not top_indices:
        return pd.DataFrame()

    results_df = products_df.iloc[top_indices].copy()
    results_df["similarity"] = cosine_sim[top_indices]

    return results_df

def recommend_personalized_by_search(query, products_df, vectorizer, tfidf_matrix, user_recs, top_n=10):
    # Search for products based on the query
    search_results = recommend_products_by_search(query, products_df, vectorizer, tfidf_matrix, top_n=50)

    if search_results.empty:
        return pd.DataFrame()

    # Filter personalized recommendations
    user_rec_ids = set(user_recs['product_id']) if 'product_id' in user_recs.columns else set()

    # Add a tag column: if it's in the personalized recommendations, label as 'Liked', otherwise ''
    search_results['tag'] = search_results['product_id'].apply(
        lambda x: "Liked" if x in user_rec_ids else ""
    )

    # Sort: prioritize 'Liked' products at the top
    search_results = search_results.sort_values(by='tag', ascending=False)

    return search_results.head(top_n)

def render_selected_product(products_df, placeholder_image, product_id, tfidf_matrix):
    product = products_df[products_df['product_id'].astype(str) == str(product_id)].squeeze()
    if product is None or (hasattr(product, "empty") and product.empty):
        st.error("Không tìm thấy sản phẩm.")
    else:
        st.markdown("## 🛒 Chi tiết sản phẩm")
        render_detail_product(product, placeholder_image)

        st.markdown("### 🔁 Sản phẩm liên quan:")
        similar_products = get_similar_products(int(product_id), products_df, tfidf_matrix)
        if similar_products is not None and not similar_products.empty:
            render_recommended_products_ralated(similar_products, placeholder_image)

        
def render_search_result(search_query, products_df, placeholder_image, vectorizer, tfidf_matrix):
    # search_query = st.session_state.get("search_query", "").strip()

    if search_query:
        # Gợi ý sản phẩm theo nội dung mô tả/tên sản phẩm
        suggested_df = recommend_products_by_search(search_query, products_df, vectorizer, tfidf_matrix, top_n=10)

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



def render_product_card_product1(row, placeholder_image):
    """Hiển thị thẻ sản phẩm (product card) từ dòng dữ liệu (row)"""
    image_path = row.get('image', '')
    if not image_path or str(image_path).strip().lower() in ['nan', 'none', 'null', '']:
        image_path = placeholder_image

    # html image
    if os.path.exists(image_path):
        base64_img = image_to_base64(image_path)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image">'
    elif str(image_path).startswith("http"):
        image_html = f'<img src="{image_path}" alt="Product Image">'
    else:
        base64_img = image_to_base64(placeholder_image)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image">'

    # create star rating
    rating = row.get('rating', 0)
    try:
        rating = float(rating)
    except:
        rating = 0
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    star_html = '⭐' * full_stars + ('✨' if half_star else '')

    # create product card
    st.markdown(f"""
    <div class="card-wrapper">
        <div class="card">
            <div class="card-img">{image_html}</div>
            <div class="card-title">
                <a href="/?product_id={row['product_id']}" style="text-decoration: none; color: inherit;">
                    {truncate_text(row.get('product_name', 'Không tên'), max_length=70)}
                </a>
            </div>
            <div class="card-id"><strong>Product_id:</strong> {row.get('product_id', 'N/A')}</div>
            <div class="card-subcat"><strong>Category:</strong> {row.get('sub_category', 'Không rõ')}</div>
            <div class="card-rating">Rating: {star_html} ({rating:.1f}/5)</div>
            <div class="card-price">{int(row.get('price', 0)):,}đ</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # make description
    with st.expander("📄 Mô tả chi tiết"):
        st.markdown(row.get('description', 'Không có mô tả'))

def render_product_card_product(row, placeholder_image):
    """Hiển thị thẻ sản phẩm với bố cục 2 cột: hình ảnh | thông tin"""
    image_path = row.get('image', '')
    if not image_path or str(image_path).strip().lower() in ['nan', 'none', 'null', '']:
        image_path = placeholder_image

    # html image
    if os.path.exists(image_path):
        base64_img = image_to_base64(image_path)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image" style="width:100%; border-radius: 8px;">'
    elif str(image_path).startswith("http"):
        image_html = f'<img src="{image_path}" alt="Product Image" style="width:100%; border-radius: 8px;">'
    else:
        base64_img = image_to_base64(placeholder_image)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" alt="Product Image" style="width:100%; border-radius: 8px;">'

    # create star rating
    rating = row.get('rating', 0)
    try:
        rating = int(rating)
    except:
        rating = 0

    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    star_html = '⭐' * full_stars + ('✨' if half_star else '')  

    # create product card
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(image_html, unsafe_allow_html=True)

    with col2:
        product_title = row.get('product_name', 'Tên sản phẩm không xác định')
        product_id = row.get('product_id', 'N/A')
        category = row.get('sub_category', 'Không rõ')
        price = row.get('price', 0)
        similarity = row.get('similarity', 0)
        rating = row.get('rating', 0)
        star_html = "⭐" * int(round(rating))

        st.markdown(f"""
        <div style='padding: 10px 0;'>
            <h5 style='margin-bottom: 5px;'>{product_title} {'<span style="color:#EE4D2D;">❤️ Yêu thích</span>' if row.get('tag') == 'Yêu thích' else ''}</h5>
            <ul style='list-style: none; padding-left: 0;'>
                <li><b>Product_id:</b> {product_id}</li>
                <li><b>Category:</b> {category}</li>
                <li><b>Rating:</b> {star_html} ({rating:.1f}/5)</li>
                <li><b>Price:</b> <span style='color: #EE4D2D; font-weight: bold;'>{int(price):,}đ</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📄 Mô tả chi tiết"):
            st.markdown(row.get('description', 'Không có mô tả'))

def get_product_card_html(row, placeholder_image):

    image_path = row.get('image', '')
    if not image_path or str(image_path).strip().lower() in ['nan', 'none', 'null', '']:
        image_path = placeholder_image

    if os.path.exists(image_path):
        base64_img = image_to_base64(image_path)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" class="product-image" />'
    elif str(image_path).startswith("http"):
        image_html = f'<img src="{image_path}" class="product-image" />'
    else:
        base64_img = image_to_base64(placeholder_image)
        image_html = f'<img src="data:image/jpeg;base64,{base64_img}" class="product-image" />'

    product_title = row.get('product_name', 'Tên sản phẩm không xác định')
    product_id = row.get('product_id', 'N/A')
    category = row.get('sub_category', 'Không rõ')
    price = row.get('price', 0)
    rating = float(row.get('rating', 0))
    description = row.get('description', 'Không có mô tả.')
    star_html = '⭐' * int(round(rating))

    # create product card
    desc_id = f"desc_{product_id}"

    return f"""
    <div class="product-card">
        <div class="image-container">{image_html}</div>
        <div class="product-info">
            <hp style='margin-bottom: 10px; font-size: 12px;'> {'<span style="color:#EE4D2D;">❤️ Collaborative Filtering</span>' if row.get('tag') == 'Yêu thích' else ''}</hp>
            <h4>{product_title}</h4>
            <button onclick="toggleDescription('{desc_id}')" class="toggle-btn">Hiện / Ẩn mô tả</button>
            <p id="{desc_id}" class="description" style="display:none;">{description}</p>
            <p><b>ID:</b> {product_id}</p>
            <p><b>Category:</b> {category}</p>
            <p><b>Rating:</b> {star_html} ({rating:.1f}/5)</p>
            <p><b>Price:</b> <span class="price">{int(price):,}đ</span></p>
        </div>
    </div>
    """

# Display recommended products
def render_recommended_products_ralated(related_products, placeholder_image):
    cols = st.columns(3)
    for i, (_, row) in enumerate(related_products.iterrows()):
        with cols[i % 3]:
            render_product_card_product(row, placeholder_image)

def render_recommended_products_by_row(related_products, placeholder_image):
    for _, row in related_products.iterrows():
        render_product_card_product(row, placeholder_image)



def render_scrollable_products_html(related_products, placeholder_image):
    cards_html = ''.join([get_product_card_html(row, placeholder_image) for _, row in related_products.iterrows()])

    full_html = f"""
    <style>
        .product-scroll-container {{
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 12px;
            background-color: #ffffff;
            font-family: "Roboto", sans-serif;
        }}
        .product-card {{
            display: flex;
            padding: 12px;
            border-bottom: 1px solid #eee;
            transition: background 0.2s ease;
        }}
        .product-card:hover {{
            background-color: #f5f5f5;
        }}
        .image-container {{
            margin-right: 15px;
        }}
        .product-image {{
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .product-info h4 {{
            margin: 0;
            font-size: 16px;
            color: #333;
        }}
        .product-info .description {{
            margin: 4px 0 6px 0;
            font-size: 13px;
            color: #666;
            font-style: italic;
        }}
        .product-info p {{
            margin: 4px 0;
            font-size: 14px;
            color: #444;
        }}
        .price {{
            color: #EE4D2D;
            font-weight: bold;
        }}
        .toggle-btn {{
            margin: 4px 0;
            padding: 4px 8px;
            font-size: 13px;
            background-color: #eee;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .toggle-btn:hover {{
            background-color: #ddd;
        }}
    </style>

    <script>
        function toggleDescription(id) {{
            var elem = document.getElementById(id);
            if (elem.style.display === "none") {{
                elem.style.display = "block";
            }} else {{
                elem.style.display = "none";
            }}
        }}
    </script>

    <div class="product-scroll-container">
        {cards_html}
    </div>
    """

    components.html(full_html, height=1000, scrolling=True)

def render_recommended_products_for_user(recommendation_products, placeholder_image):
    cols = st.columns(3)
    for i, (_, row) in enumerate(recommendation_products.iterrows()):
        label = "✅ Rất phù hợp với bạn" if i < 3 else "💡 Có thể bạn sẽ thích"
        
        with cols[i % 3]:
            st.markdown(f"<div style='font-weight: bold; color: #64B149;'>{label}</div>", unsafe_allow_html=True)
            render_product_card_product(row, placeholder_image)

def recommend_personalized_related_products(product_id, products_df, tfidf_matrix, user_recs, top_n=6):
    # Get similar products
    content_similar = get_similar_products(product_id, products_df, tfidf_matrix, top_n=50)

    if content_similar.empty:
        return pd.DataFrame()

    # Make personalized recommendations
    user_rec_ids = set(user_recs['product_id'])

    # Copy the DataFrame
    content_similar = content_similar.copy()

    # Add a column to indicate if the product is in the personalized recommendations
    content_similar['is_user_rec'] = content_similar['product_id'].apply(lambda x: x in user_rec_ids)

    # Add a tag column
    content_similar['tag'] = content_similar['is_user_rec'].apply(lambda x: "Yêu thích" if x else "")

    # Sort by is_user_rec
    content_similar = content_similar.sort_values(by='is_user_rec', ascending=False)

    # Return top_n
    return content_similar.head(top_n).drop(columns=['is_user_rec'])

# Render detail product
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
    # get rating
    rating = product.get('rating', 0)
    try:
        rating = float(rating)
    except:
        rating = 0
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    star_html = '⭐' * full_stars + ('✨' if half_star else '')
    st.markdown(f"""
    <div class="product-detail-container">
        <div class="product-detail-left">
            {image_html}
        </div>
        <div class="product-detail-right">
            <div class="product-name">{product.get('product_name', 'Không tên')}</div>
            <div class="product-rating">{star_html} ({rating:.1f}/5)</div>
            <div class="product-price">{int(product.get('price', 0)):,}đ</div>
            <div class="product-category">Category: {product.get('sub_category', 'Không rõ')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Display information
    with st.expander("📝 Mô tả chi tiết"):
        st.markdown(f"""
        <div class="product-description-content">
            {product.get('description', 'Không có mô tả.')}
        </div>
        """, unsafe_allow_html=True)

def filter_products(products_df, category, price_range, min_rating):
    # Filter by category
    if category and category != "-- Tất cả --":
        products_df = products_df[products_df['sub_category'] == category]

    # Filter by price
    if price_range:
        min_price, max_price = price_range
        products_df = products_df[(products_df['price'] >= min_price) & (products_df['price'] <= max_price)]

    # Filter by rating
    if min_rating is not None:
        products_df = products_df[products_df['rating'] >= min_rating]

    return products_df


def get_similar_products(product_id, products_df, tfidf_matrix, top_n=10):
    # Check if product exists
    if product_id not in products_df['product_id'].values:
        return pd.DataFrame()

    # Get index of the product
    idx_list = products_df.index[products_df['product_id'] == product_id].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]

    # Check TF-IDF matrix bounds
    if idx >= tfidf_matrix.shape[0]:
        return pd.DataFrame()

    # Filter products in the same category
    target_category = products_df.loc[idx, 'sub_category']
    same_category_df = products_df[products_df['sub_category'] == target_category].copy()
    same_category_indices = same_category_df.index.tolist()

    # Get cosine similarity
    product_vec = tfidf_matrix[idx]
    cosine_scores = cosine_similarity(product_vec, tfidf_matrix).flatten()
    filtered_scores = [(i, cosine_scores[i]) for i in same_category_indices if i != idx]
    filtered_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top N similar indices
    top_indices = [i for i, score in filtered_scores[:top_n]]

    return products_df.loc[top_indices].copy()

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
            continue  # pass if prediction fails

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_products = pd.DataFrame(predictions[:top_n], columns=['product_id', 'EstimateScore'])
    return top_products

def render_highlighted_product(product, highlight_ids):
    """Render sản phẩm và tô màu nếu là sản phẩm được gợi ý cho người dùng"""
    product_name = product['product_name']
    product_id = product['product_id']
    
    # Check if the product ID is in the highlight list
    if product_id in highlight_ids:
        # If it is, highlight the product name
        product_name = f"<span style='color: #FF6347; font-weight: bold;'>{product_name}</span>"
    
    # Render the product
    st.markdown(f"### {product_name}")
    st.write(f"{product['price']}đ")
    st.write(f"Mô tả: {product['description']}")

def show_selected_product(products_df, product_id, placeholder_image, tfidf_matrix=None, recommend_products_latest=None):
    product = products_df[products_df['product_id'].astype(str) == str(product_id)].squeeze()
    if product.empty:
        st.error("Không tìm thấy sản phẩm.")
    else:
        render_detail_product(product, placeholder_image)

def recommend_related_products(product_id, products_df, tfidf_matrix, top_n=6):
    return get_similar_products(int(product_id), products_df, tfidf_matrix, top_n=top_n)