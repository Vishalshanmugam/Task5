# 🧠 Consumer Complaints Classification using NLP & Machine Learning

## 📘 Overview

This project focuses on analyzing and classifying consumer complaints from the **Consumer Financial Protection Bureau (CFPB)** dataset.  
Each complaint includes a **text narrative** written by a consumer describing their issue with a financial product (e.g., mortgage, debt collection, credit reporting).  

The goal is to use **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to automatically classify these complaints into their respective **product categories**.

---

## 🎯 Objectives

- Clean and preprocess raw textual data.  
- Convert text into numerical features using **TF-IDF** (*Term Frequency-Inverse Document Frequency).  
- Train multiple machine learning models to classify complaints.  
- Evaluate model accuracy and compare performance.  
- Build a simple prediction function for new complaint texts.

---

## ⚙️ Workflow Summary

The project is structured in a series of notebook cells, each performing a specific step:

1. **Environment Setup** – Install required packages (`wordcloud`, `nltk`, `scikit-learn`, `seaborn`, etc.).  
2. **Data Loading** – Load the CFPB dataset (`consumer_complaints.csv`) with limited rows to prevent memory issues.  
3. **Exploratory Data Analysis (EDA)** – Inspect columns, check missing values, and visualize complaint distribution.  
4. **Text Preprocessing** – Clean complaint narratives:
   - Lowercasing  
   - Removing punctuation, numbers, and stopwords  
   - Tokenizing and lemmatizing text  
5. **Feature Extraction** – Convert cleaned text into **TF-IDF vectors** for model training.  
6. **Model Training & Evaluation** – Use various ML models (Logistic Regression, SVM, Naive Bayes, Random Forest) to classify complaints.  
7. **Performance Visualization** – Compare accuracy scores across models.  
8. **Prediction** – Build a function that predicts the product category of a new complaint.

---

## 🧩 Libraries and Their Roles

### 📊 **Data Handling**
| Library | Purpose |
|----------|----------|
| `pandas` | For data manipulation and reading the CSV file. |
| `numpy` | For numerical operations and array handling. |

---

### 🧹 **Text Processing (NLP)**
| Library | Purpose |
|----------|----------|
| `re` | Regular expressions for cleaning unwanted characters. |
| `nltk` | Core NLP toolkit for tokenization, stopword removal, and lemmatization. |
| `WordNetLemmatizer` | Converts words to their base form (e.g., “running” → “run”). |
| `stopwords` | Removes common non-informative words like “the”, “is”, etc. |
| `word_tokenize` | Splits sentences into individual words for processing. |

---

### 🤖 **Machine Learning**
| Library | Purpose |
|----------|----------|
| `scikit-learn` | Provides ML models, preprocessing, and evaluation tools. |
| `TfidfVectorizer` | Converts text into numerical features (term importance). |
| `LogisticRegression`, `SVC`, `MultinomialNB`, `RandomForestClassifier` | Different classification algorithms to compare performance. |
| `train_test_split` | Divides data into training and testing subsets. |
| `classification_report`, `accuracy_score` | Model performance metrics. |

---

### 🎨 **Visualization**
| Library | Purpose |
|----------|----------|
| `matplotlib` | Basic plotting for distributions and results. |
| `seaborn` | Enhanced, styled visualizations for easier interpretation. |
| `wordcloud` *(optional)* | Displays frequent words in complaint texts (visual insight). |

---

## 🧠 Machine Learning Models Used

| Model | Description | Strengths |
|--------|--------------|------------|
| **Logistic Regression** | A linear model for binary/multi-class classification. | Fast and interpretable. |
| **SVM (Support Vector Machine)** | Finds an optimal separating boundary between classes. | Works well with high-dimensional data like TF-IDF. |
| **Naive Bayes (MultinomialNB)** | Probabilistic model assuming feature independence. | Performs well with text data. |
| **Random Forest** | Ensemble of decision trees. | Robust and handles complex data patterns. |

Each model is trained and evaluated on a train/test split of complaint narratives.

---

## 📈 Evaluation Metrics

The project uses:
- **Accuracy** — overall correctness of predictions.
- **Precision, Recall, F1-score** — from `classification_report()` for detailed per-class analysis.

After training, model accuracies are compared visually in a bar chart.

---

## 💬 Prediction Function

Once the best-performing model (e.g., Random Forest) is chosen, a helper function allows quick prediction:


```python
predict_category("I was charged an unexpected fee on my credit card.")

![Project Screenshot](https://github.com/Vishalshanmugam/Task5/blob/main/Screeenshot/screenshot.png)
```

## Output 

### Complaint categories

![Project Screenshot](https://github.com/Vishalshanmugam/Task5/blob/main/Screeenshot/screenshot2.png)

### Logitstic regression (precision and accuracy)

![Project Screenshot](https://github.com/Vishalshanmugam/Task5/blob/main/Screeenshot/screenshot3.png)

### Model accuracy as graphs

![Project Screenshot](https://github.com/Vishalshanmugam/Task5/blob/main/Screeenshot/screenshot4.png)

The whole Task is uploaded as Task5.ipynb jupyter notebook in the same repository
