# AI--Tweets-Disaster-Management-system
“An AI-powered NLP system that classifies tweets in real-time as disaster-related or not, using machine learning and deep learning (LSTM). Helps monitor crises via social media by preprocessing text, training multiple models, and providing accurate disaster detection to support timely response.”

# 🌐 AI for Real-Time Disaster Management

This project is an intelligent NLP system designed to classify tweets as disaster-related or not, using a combination of machine learning (Logistic Regression, Naive Bayes, SVM) and deep learning (LSTM) techniques. The aim is to support real-time disaster monitoring and early warning systems by analyzing social media data.

---

## 📑 Project Description

In times of crisis, timely detection of disaster-related information is crucial for effective response and resource allocation. This project leverages Natural Language Processing (NLP) to preprocess tweets, extract relevant features, and classify them with high accuracy. It demonstrates an end-to-end pipeline: data acquisition, cleaning, visualization, classical ML training, deep learning with LSTM, and a prediction interface for real-time tweet classification.

Key highlights:
- NLP preprocessing: tokenization, stopword removal, lemmatization
- Feature extraction: TF-IDF vectorization
- Machine learning: Logistic Regression, Naive Bayes, Support Vector Machine
- Deep learning: LSTM with embedding for sequential context understanding
- Visualizations: class distribution, accuracy plots
- Ready for extension to real-time streaming and multilingual support

---

## 🗂️ Project Structure
📁 data/ # Dataset files (train_clean.csv, etc.)
📁 notebooks/ # Jupyter/Colab notebooks
📁 models/ # Saved ML/DL models
📁 scripts/ # Python scripts for preprocessing, training, prediction
📁 outputs/ # Plots, reports, evaluation metrics
📄 README.md # Project overview and instructions

yaml
Copy
Edit

---

## ⚙️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn, TensorFlow, Keras
- **Environment:** Google Colab (with GPU), Jupyter Notebook

---

## 🚀 How to Run

1️⃣ **Clone this repository**


git clone https://github.com/yourusername/AI-Disaster-Tweet-Classifier.git
cd AI-Disaster-Tweet-Classifier

2️⃣ **Install required dependencies**

pip install -r requirements.txt

3️⃣ **Download dataset**

Update your Kaggle API credentials and download the dataset using KaggleHub or place train_clean.csv in the data/ folder.

4️⃣ **Run the notebook**

Open the Jupyter notebook or Google Colab and run step-by-step:

Data preprocessing

Feature extraction

Train ML models

Train LSTM model

Evaluate performance

Test prediction function

🧩 **Future Scope**
Multilingual classification with transformer models (e.g., mBERT, XLM-RoBERTa)

Real-time deployment with APIs, stream processing, and cloud services

Geolocation extraction and integration with live disaster dashboards

Sentiment and emotion detection for crisis severity assessment

Continuous model improvement through user feedback loops

🤝 **Contributing**
Pull requests and suggestions are welcome! Feel free to fork this repo and improve it.

📜 **License**
This project is open-source and free to use under the MIT License.

✨ **Acknowledgements**
Dataset: Kaggle - NLP with Disaster Tweets

Developed as a part of an academic project to demonstrate practical applications of NLP and AI for disaster management.




