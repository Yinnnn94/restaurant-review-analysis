import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import os

data = pd.read_csv(r'reviews.csv')
business_names = list(data['business_name'].unique())
image_dir = "dataset"
photo_cat = ['indoor atmosphere',  'outdoor atmosphere', 'taste', 'menu']
images_per_row = 3

# First summarize
def summarize(data, business_name):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    subjective_words = ["I", "me", "my", "mine", "we", "us", "our", "ours"]
    subjective_tokens = tokenizer.convert_tokens_to_ids(subjective_words)
    summarized_texts = []
    for text in data[data['business_name'] == business_name]['text']:
        inputs = tokenizer("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
        if len(inputs['input_ids'][0]) > 50:
            summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4,
                                         bad_words_ids=[[token] for token in subjective_tokens], early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summarized_texts.append(summary)
        else:
            summarized_texts.append(text)
    return summarized_texts

# Final summarize
def second_summarize(summarized_texts):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    combined_text = " ".join(summarized_texts)
    inputs = tokenizer("summarize: " + combined_text, return_tensors='tf', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return final_summary

# Extract keywords
def extract_keywords(texts):
    nlp = spacy.load('en_core_web_sm')
    combined_text = " ".join(texts)
    doc = nlp(combined_text)
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'ADJ')]
    keyword_freq = Counter(keywords)
    return keyword_freq.most_common(5)

# Plot bar chart and word cloud
def plot_keywords(business_name, most_common_keywords):
    # Bar chart for ratings
    counts = data[data['business_name'] == business_name]['rating'].value_counts().sort_index()
    bar_fig = go.Figure([go.Bar(x=counts.index, y=counts.values, marker_color='indianred')])
    bar_fig.update_layout(title_text='Rating Bar Graph', xaxis_title='Rating', yaxis_title='Count')

    # Word cloud
    word_freq_dict = dict(most_common_keywords)
    wc = WordCloud(width=500, height=300, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate_from_frequencies(word_freq_dict)
    wc_img = wc.to_image()
    return bar_fig, wc_img

def display_img(bussines_name, selected_category):
    # Display selected images
    if 'selected_category' in locals():
        st.subheader(selected_category.capitalize())
        image_files = data[data['business_name'] == bussines_name]['photo'].dropna().values
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
        # Summary
        summarized_texts = summarize(data, option)
        final_summary = second_summarize(summarized_texts)
        most_common_keywords = extract_keywords(summarized_texts)
        st.header('Review Summary')
        st.write(final_summary)
        bar_fig, wc_img = plot_keywords(option, most_common_keywords)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_fig)
        with col2:
            st.image(wc_img, use_column_width=True)
        st.header("Picture Gallery")
        display_img(option, selected_category)

        st.success('Done!')