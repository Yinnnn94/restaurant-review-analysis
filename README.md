## Restaurant Review Analysis 🍽️
This project analyzes restaurant reviews using NLP and displays the results with interactive visualizations.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-project.git
    cd your-project
    ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews).

3. pip install -r requirements.txt

4. Create a directory named `data` in the root of your project and place the downloaded CSV file and images into this directory.

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

5. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

Select a restaurant from the dropdown to view a summary of its reviews and a gallery of related images.
