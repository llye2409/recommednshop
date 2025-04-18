import streamlit as st

def about_team():
    st.title("🧑 About the Team")
    st.markdown("""
    The team consists of a group of students from different fields, currently studying the **Data Science & Machine Learning** course at the **Computer Science Center - University of Science, Ho Chi Minh City**.

    This project is part of our learning journey and practical application. We combine skills and experience from various fields to build this product.
    """)

    st.markdown("---")
    st.subheader("🔧 Tasks")

    st.markdown("""
        ### 👤 Nguyễn Văn Duy
        - Data collection and product data processing  
        - Data cleaning and normalization of attributes  
        - Initial descriptive statistical analysis  
        - Support in creating data reports

        ### 👤 Phạm Vũ An
        - Analyzing user behavior based on rating data  
        - Data visualization (bar charts, wordclouds, heatmaps...)  
        - Extracting insights from the analyzed data  
        - Designing and presenting reports

        ### 👤 Nguyễn Văn Gulist
        - Designing and building the Streamlit interface  
        - Integrating analyses into the web interface  
        - Optimizing processing performance and data display  
        - Managing project structure & deployment
        """)

    st.markdown("---")

    st.write("""
    ### 🚀 **Contact**
    If you have any questions or would like to collaborate with us, feel free to reach out via email:  
    📧 **contact@teamstreamlit.com**

    ### 💬 **Follow us**
    - [LinkedIn](https://www.linkedin.com)
    - [Twitter](https://twitter.com)
    - [GitHub](https://github.com)
    """)
