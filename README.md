# spam-detection
this is my first visible project enjoy!
# 📧 Spam Detection Model with Scikit-Learn

## 📌 Project Overview
This project aims to build a **Spam Detection Model** using **Scikit-Learn** and a dataset from **Kaggle**. The model classifies emails as **spam** or **ham (not spam)** based on text content. 

## 📂 Dataset
- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: A dataset containing **5,574 messages**, labeled as `ham` (not spam) or `spam`.
- **Format**: CSV with two columns: 
  - `label`: `ham` or `spam`
  - `message`: The SMS content

## 🛠 Technologies Used
- **Python**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **NLTK** (for text preprocessing)
- **Matplotlib & Seaborn** (for visualization)

## 🚀 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the dataset from Kaggle** and place it in the `data/` folder.
5. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## 🔍 Model Training & Evaluation
### 1️⃣ Data Preprocessing
- Convert text to lowercase
- Remove punctuation & stopwords
- Tokenization & Lemmatization
- Convert text to numerical features using **TF-IDF Vectorizer**

### 2️⃣ Model Selection
Tested multiple models, including:
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Random Forest Classifier

**Best Model:** `Multinomial Naive Bayes` with **~98% accuracy**

### 3️⃣ Model Evaluation
Metrics used:
- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

## 📊 Results
| Model | Accuracy |
|------------|-----------|
| Naive Bayes | **98.3%** |
| Logistic Regression | 96.5% |
| SVM | 97.8% |

## 📎 Usage
To predict whether a new SMS is spam or not, use:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_spam(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return 'Spam' if prediction[0] == 'spam' else 'Not Spam'

print(predict_spam("Congratulations! You've won a free iPhone! Click here to claim."))
```

## 🏆 Future Improvements
- Try **Deep Learning models (LSTMs)**
- Improve preprocessing with **BERT embeddings**
- Deploy the model using **Flask or FastAPI**

## 👩‍💻 Author
- **noussaiba kerrache**
- **GitHub**: [nousskrr](https://github.com/nousskrr)
- **LinkedIn**: [noussaiba kerrache](https://www.linkedin.com/in/krr-noussaiba-a03892312/)

## ⭐ Contributions & Feedback
Feel free to **fork**, **open issues**, or **submit PRs**! Feedback is welcome. 😊
