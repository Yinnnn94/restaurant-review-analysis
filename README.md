## Restaurant Review Analysis 🍽️
This project analyzes restaurant reviews using NLP and displays the results with interactive visualizations.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-project.git
    cd your-project
    ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews).

3. Setup the module
     ```bash
    pip install -r requirements.txt
    ```

5. Create a directory named `data` in the root of your project and place the downloaded CSV file and images into this directory.

    ```
    your-project/
    ├── app.py
    ├── README.md
    ├── .gitignore
    ├── requirements.txt
    └── data/
        ├── reviews.csv
        └── dataset/
    ```

6. Run the application:
    ```bash
    streamlit run app.py
    ```

7. UI display:
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/Yinnnn94/restaurant-review-analysis/blob/main/UI%20display/UI%20display1.png" width="45%" alt="UI display 1">
    <img src="https://github.com/Yinnnn94/restaurant-review-analysis/blob/main/UI%20display/UI%20display2.png" width="45%" alt="UI display 2">
</div>

- The picture on the right shows the restaurant summary review along with the high-rate and low-rate Wordcloud.
- The picture on the left displays the selected image.

## Usage

Select a restaurant from the dropdown to view a summary of its reviews and a gallery of related images.
