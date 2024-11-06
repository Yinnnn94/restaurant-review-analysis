import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from wordcloud import WordCloud
from transformers import pipeline
from PIL import Image
import os

data = pd.read_csv(r'reviews.csv')
business_names = list(data['business_name'].unique())
image_dir = "dataset"
photo_cat = ['indoor atmosphere', 'outdoor atmosphere', 'taste', 'menu']
images_per_row = 3

def summarize(data, business_name):
    all_comment = data[data['business_name'] == business_name]['text'].values
    combined_text = " ".join(all_comment)
    summarizes = pipeline("summarization", model="facebook/bart-large-cnn")  # model
    summarizes_list = summarizes(combined_text, max_length=len(combined_text) * 0.8, min_length=30, do_sample=False)
    return summarizes_list[0]['summary_text']

# Plot bar chart and word cloud
def plot_keywords(business_name):
    # Bar chart for ratings
    counts = data[data['business_name'] == business_name]['rating'].value_counts().sort_index()
    bar_fig = go.Figure([go.Bar(x=counts.index, y=counts.values, marker_color='indianred')])
    bar_fig.update_layout(title_text='Rating Bar Graph', xaxis_title='Rating', yaxis_title='Count')

    # High rating Word Cloud
    high_data = data[(data['business_name'] == business_name) & (data['rating'] > 3)]
    high_comment = " ".join(high_data['text'].values)
    high_wc = WordCloud(width=500, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(high_comment)
    high_wc_img = high_wc.to_image()

    # Low rating Word Cloud
    low_data = data[(data['business_name'] == business_name) & (data['rating'] < 3)]
    low_comment = " ".join(low_data['text'].values)
    low_wc = WordCloud(width=500, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(low_comment)
    low_wc_img = low_wc.to_image()

    return bar_fig, high_wc_img, low_wc_img

def display_img(business_name, selected_category):
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
option = st.selectbox("Which restaurant do you want to analyze?", business_names, 
                    index=None, placeholder="Select restaurant")

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
        st.header(f'Comment of {option}')
        st.write(summary)

        bar_fig, high_wc_img, low_wc_img = plot_keywords(option)

        # Display the bar chart and word clouds for high and low ratings
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_fig)
        with col2:
            st.image(low_wc_img, caption="Low Rating Word Cloud", use_column_width=True)
            st.image(high_wc_img, caption="High Rating Word Cloud", use_column_width=True)

        st.header("Picture Gallery")
        display_img(option, selected_category)

        st.success('Done!')
