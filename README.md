# Fake News Detector

A deep learning-based web app and notebook for detecting fake news using Natural Language Processing (NLP) and LSTM neural networks.

## Features

- **Data Cleaning & EDA:** Automated preprocessing, stopword removal, and exploratory data analysis on real and fake news datasets.
- **Visualization:** Data distribution, word clouds, and class balance visualizations using Matplotlib, Seaborn, and Plotly.
- **Modeling:** Bidirectional LSTM neural network for binary text classification, achieving high accuracy.
- **Web App:** Interactive Streamlit app for real-time fake news prediction.
- **Reproducible Notebook:** Step-by-step Jupyter Notebook for data science workflow and experimentation.

## Project Structure

```
FakeNewsDetector-ML_Project/
│
├── FakeNewsClassification.ipynb   # Jupyter Notebook with full workflow
├── app.py                        # Streamlit web app
├── True.csv                      # Real news dataset
├── Fake.csv                      # Fake news dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FakeNewsDetector-ML_Project.git
cd FakeNewsDetector-ML_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Explore the Jupyter Notebook

Open `FakeNewsClassification.ipynb` in Jupyter or VS Code to review the full data science workflow.

## Usage

- **Web App:** Paste any news article text into the Streamlit sidebar and get an instant prediction (Real or Fake).
- **Notebook:** Follow the step-by-step workflow for data loading, cleaning, EDA, modeling, and evaluation.

## Model

- **Architecture:** Bidirectional LSTM with embedding layer and dense output.
- **Training:** Trained on 40,000+ news articles, achieving over 95% accuracy on the test set.

## Screenshots

*(Add screenshots of the app and notebook here if desired)*

## License

This project is licensed under the MIT License.

---

**Author:** [AlienMinus]  
**Contact:** [dasmanasranjan2005@gmail.com]