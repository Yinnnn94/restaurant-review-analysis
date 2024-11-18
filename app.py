import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from wordcloud import WordCloud
from transformers import pipeline
from PIL import Image
import os
import multidict as multidict
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 加載數據
data = pd.read_csv(r'reviews.csv')
business_names = list(sorted(data['business_name'].unique()))
image_dir = "dataset"
photo_cat = ['indoor atmosphere', 'outdoor atmosphere', 'taste', 'menu']
images_per_row = 3

# 初始化模型
summarize_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# 摘要函數
def summarize(data, business_name):
    all_comment = data[data['business_name'] == business_name]['text'].dropna().values
    combined_text = " ".join(all_comment)
    try:
        summary_list = summarize_pipeline(
            combined_text, max_length=min(len(combined_text.split()) * 0.8, 512), 
            min_length=30, do_sample=False
        )
        return summary_list[0]['summary_text']
    except Exception as e:
        return "Unable to summarize the text due to length or data issues."

# 詞頻字典生成
def getFrequencyDictForText(sentence):
    STOPWORDS = set([
        "you", "I", "he", "she", "it", "we", "they", "and", "or", "but", "the", "a", "an", "of", "for",
        "to", "in", "on", "with", "at", "is", "was", "are", "were", "be", "been", "by", "that", "this",
        "those", "these", "not", "as", "from", "they", "us", "our", "me", "my", "your", "their", "its",
        "will", "would", "can", "could", "should", "about"
    ])
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}
    cleaned_text = re.sub(r"[^\w\s]", "", sentence).lower()

    for word in cleaned_text.split():
        if word in STOPWORDS or len(word) <= 1:
            continue
        val = tmpDict.get(word, 0)
        tmpDict[word] = val + 1

    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])

    return fullTermsDict
def generate_wordcloud(text):
        if text.strip():
            wc = WordCloud(width=500, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA")
            return wc.generate(text).to_image()
        return None

def extract_keywords(rating_filter, business_name):
    filtered_text = " ".join(data[(data['business_name'] == business_name) & rating_filter]['text'].dropna())
    return filtered_text

# 繪圖函數
def plot_keywords(business_name):
    counts = data[data['business_name'] == business_name]['rating'].value_counts().sort_index()
    bar_fig = go.Figure([go.Bar(x=counts.index, y=counts.values, marker_color='indianred')])
    bar_fig.update_layout(title_text='Rating Bar Graph', xaxis_title='Rating', yaxis_title='Count')

    
    high_wc_img = generate_wordcloud(extract_keywords(data['rating'] > 3, business_name))
    low_wc_img = generate_wordcloud(extract_keywords(data['rating'] < 3, business_name))

    return bar_fig, high_wc_img, low_wc_img

# 圖片顯示函數
def display_images(business_name, selected_category):
    # Display selected images
    if 'selected_category' in locals():
        st.subheader(selected_category.capitalize())
        image_files = data[data['business_name'] == business_name]['photo'].dropna().values
        filtered_images = [img for img in image_files if selected_category in img]
        num_images = len(filtered_images)
        if num_images > 0:
            for i in range(0, num_images, images_per_row):
                cols = st.columns(min(images_per_row, num_images - i))
                for j, col in enumerate(cols):
                    if i + j < num_images:
                        img_path = os.path.join(image_dir, filtered_images[i + j])
                        img = Image.open(img_path)
                        col.image(img, use_column_width=True)

# Streamlit UI
st.title('Restaurant Review Analysis')
option = st.selectbox("Which restaurant do you want to analyze?", business_names, index=None, placeholder="Select restaurant")

# Picture Gallery
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button(photo_cat[0]):
        selected_category = photo_cat[0]
with col2:
    if st.button(photo_cat[1]):
        selected_category = photo_cat[1]
with col3:
    if st.button(photo_cat[2]):
        selected_category = photo_cat[2]
with col4:
    if st.button(photo_cat[3]):
        selected_category = photo_cat[3]

if option and 'selected_category' in locals():
    with st.spinner('Please wait...'):
        summary = summarize(data, option)
        st.header(f'Summarize Comment of {option}')
        st.write(summary)

        bar_fig, high_wc_img, low_wc_img = plot_keywords(option)

        # Display the bar chart and word clouds for high and low ratings
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_fig)
        with col2:
            st.image(low_wc_img, caption="Low Rate Word Cloud", use_column_width=True)
            st.image(high_wc_img, caption="High Rate Sentiment Word Cloud", use_column_width=True)

        st.header("Picture Gallery")
        display_images(option, selected_category)

        st.success('Done!')