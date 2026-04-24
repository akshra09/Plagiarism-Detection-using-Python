# 📄 Plagiarism Detection Using Python

## Overview

This project implements a **Plagiarism Detection System** using **Natural Language Processing (NLP)** techniques in Python. It analyzes multiple text documents (for example, student assignments), transforms them into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**, and computes **Cosine Similarity** scores to detect possible plagiarism.

In addition to similarity analysis, the project includes **WordCloud visualizations** to help inspect overlapping content patterns visually.

---

## 🚀 Features

* Detects textual similarity between multiple documents
* Uses **TF-IDF Vectorization** for feature extraction
* Measures plagiarism using **Cosine Similarity**
* Performs pairwise comparison across all documents
* Flags highly similar document pairs
* Generates **WordCloud visualizations** for exploratory analysis
* Lightweight and beginner-friendly implementation

---

## 🧠 Project Workflow

The plagiarism detection pipeline follows these steps:

```text
Text Documents (.txt)
      ↓
Read and Load Files
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity Computation
      ↓
Pairwise Plagiarism Detection
      ↓
Threshold-Based Suspicion Flagging
      ↓
WordCloud Visualization
```

---

## 📂 Project Structure

```bash
Plagiarism-Detection-Using-Python/
│
├── Plagiarism_Detection_Using_Python.ipynb
├── document 1.txt
├── document 2.txt
├── document 3.txt
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Jupyter Notebook
* Scikit-learn
* Matplotlib
* WordCloud
* NLP (TF-IDF)

---

## 📌 Core Concepts Used


### 1. TF-IDF Vectorization
TF-IDF converts text into numerical vectors to evaluate word importance.

#### Term Frequency (TF)
Measures how frequently a word appears in a document:

$$TF(t,d) = \frac{\text{Number of times term } t \text{ appears}}{\text{Total terms in document}}$$

#### Inverse Document Frequency (IDF)
Reduces the weight of common words that appear across many documents:

$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

**Where:**
* **$N$** = Total number of documents
* **$df(t)$** = Number of documents containing term $t$

---

#### TF-IDF Score
The final weight is calculated by multiplying these two metrics:

$$TF\text{-}IDF = TF \times IDF$$

---

## 2. Cosine Similarity
Used to measure the logical similarity between two document vectors by calculating the cosine of the angle between them:

$$Similarity(A,B) = \frac{A \cdot B}{\|A\| \|B\|}$$

**Interpretation of Results:**
* **1.0** → Identical documents (vectors point in the same direction).
* **0.0** → Completely different (vectors are orthogonal).
* **0.5+** → Potential plagiarism (depending on your defined threshold).

---

# 🧾 Implementation Breakdown

## Importing Libraries

```python
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
```

### Purpose

* `os` → file handling
* `matplotlib` → visualization
* `TfidfVectorizer` → convert text to vectors
* `cosine_similarity` → compare documents
* `WordCloud` → visual analysis

---

## Reading Student Documents

```python
student_file = [file for file in os.listdir() if file.endswith('.txt')]
student_docs = [open(file).read() for file in student_file]
```

This automatically loads all `.txt` files from the directory.

---

## Creating TF-IDF Vectors

```python
def create_tfidf_vectors(docs):
    return TfidfVectorizer().fit_transform(docs).toarray()
```

Each document becomes a numerical vector representation.

---

## Computing Similarity

```python
def calc_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1, vector2])
```

Computes similarity score between two document vectors.

---

## Pairwise Plagiarism Detection

```python
def find_plagiarism():
    plagiarism_results = set()

    for student_a_file, student_a_vec in doc_filename_pairs:
        remaining_pairs = doc_filename_pairs.copy()

        current_index = remaining_pairs.index(
            (student_a_file, student_a_vec)
        )

        del remaining_pairs[current_index]

        for student_b_file, student_b_vec in remaining_pairs:
            similarity_score = calc_cosine_similarity(
                student_a_vec,
                student_b_vec
            )[0][1]

            sorted_filenames = sorted(
                (student_a_file, student_b_file)
            )

            plagiarism_results.add(
                (
                  sorted_filenames[0],
                  sorted_filenames[1],
                  similarity_score
                )
            )

    return plagiarism_results
```

---

## Example Output

```text
('document 1.txt', 'document 2.txt', 0.84)
('document 1.txt', 'document 3.txt', 0.22)
('document 2.txt', 'document 3.txt', 0.79)
```

Interpretation:

* 84% similarity → possible plagiarism
* 22% similarity → low overlap
* 79% similarity → suspicious similarity

---

# ☁️ WordCloud Visualization

## Generate WordCloud

```python
def generate_word_cloud(document_text, filename):
    wordcloud = WordCloud(
        width=800,
        height=400
    ).generate(document_text)

    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
```

## Why WordCloud?

WordClouds help visually inspect:

* Repeated keywords
* Common phrases
* Content overlap
* Suspicious vocabulary duplication

---

## Threshold-Based Suspicion Check

```python
if result[2] >= 0.5:
    generate_word_cloud(...)
```

Documents above 50% similarity are flagged.

You can tune thresholds:

| Similarity Score | Interpretation       |
| ---------------- | -------------------- |
| 0 - 0.30         | Low Similarity       |
| 0.30 - 0.50      | Moderate Overlap     |
| 0.50 - 0.70      | Potential Plagiarism |
| 0.70+            | High Suspicion       |

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/plagiarism-detection-python.git
cd plagiarism-detection-python
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install scikit-learn matplotlib wordcloud
```

---

## Run Notebook

```bash
jupyter notebook
```

Open:

```text
Plagiarism_Detection_Using_Python.ipynb
```

---

# 📊 Complexity Analysis

## Time Complexity

Pairwise comparison:

[
O(n^2)
]

Where:

* n = number of documents

Every document is compared against others.

---

## Space Complexity

TF-IDF Matrix:

[
O(n \times m)
]

Where:

* n = documents
* m = vocabulary size

---

# 🔍 Limitations

Current approach may struggle with:

* Paraphrased plagiarism
* Synonym substitution
* Semantic plagiarism
* Sentence reordering
* Cross-language plagiarism

Because TF-IDF captures lexical overlap more than semantic meaning.

---

# 🚀 Possible Improvements

Future enhancements:

## Semantic Similarity with Embeddings

Use:

* Sentence Transformers
* BERT embeddings
* Word2Vec
* Doc2Vec

Instead of pure TF-IDF.

---

## Add Fingerprinting / Shingling

Use:

* n-grams
* MinHash
* Winnowing algorithms

For stronger plagiarism detection.

---

## Build Web App

Convert into:

* Streamlit app
* Flask API
* Gradio interface

Upload assignments and get plagiarism reports.

---

## Generate Similarity Heatmaps

Possible extension:

```python
import seaborn as sns
```

Create a document-to-document similarity matrix.

---

# 📈 Potential Use Cases

This project can be adapted for:

* Academic plagiarism detection
* Assignment similarity checks
* Content duplication analysis
* Research paper overlap detection
* Resume similarity screening

---

## Sample Requirements File

```txt
scikit-learn
matplotlib
wordcloud
```

---

# 🏆 Learning Outcomes

This project demonstrates the practical use of:

* NLP preprocessing
* Feature engineering with TF-IDF
* Similarity metrics
* Unsupervised text analysis
* Visualization for interpretability

---


## 🤝 Contributing

Pull requests and improvements are welcome.

If you'd like to extend this project using BERT or transformer-based plagiarism detection, feel free to fork it.

---

## 📜 License

MIT License

---

