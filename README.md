# NLP-mini-porject-W2VEC
# Game of Thrones Word2Vec — NLP Mini Project

A Natural Language Processing mini project that uses **Word2Vec** to learn meaningful word representations from the *A Song of Ice and Fire / Game of Thrones* book text.

The project demonstrates text preprocessing, tokenization, Word2Vec model training, word similarity analysis, and visualization of learned word embeddings.

## 🚀 Project Overview

In this project, I trained a Word2Vec model on the Game of Thrones book text to understand how words are represented in a continuous vector space.

The model learns relationships between words based on their surrounding context.

For example, after training, words such as:

* `king`
* `queen`
* `prince`
* `castle`
* `dragon`

can develop meaningful relationships because they frequently appear in related contexts throughout the books.

## 🧠 What is Word2Vec?

Word2Vec is a popular NLP technique that converts words into numerical vectors called **word embeddings**.

Instead of representing a word as a simple ID, Word2Vec represents it as a vector containing numerical values.

For example:

```text
king → [0.21, -0.43, 0.72, ...]
queen → [0.18, -0.39, 0.69, ...]
```

Words with similar meanings or contextual usage tend to have vectors that are closer together.

### Word2Vec Architectures

Word2Vec mainly uses two approaches:

1. **CBOW — Continuous Bag of Words**

   * Predicts a target word from surrounding context words.

2. **Skip-Gram**

   * Predicts surrounding context words from a target word.

In this project, Word2Vec is used to learn embeddings from the Game of Thrones text corpus.

## 📚 Dataset

The dataset consists of text from the Game of Thrones / *A Song of Ice and Fire* books.

The raw text is processed into sentences and then tokenized before being passed to the Word2Vec model.

## 🔄 NLP Pipeline

```text
Raw Book Text
      ↓
Text Cleaning
      ↓
Sentence Splitting
      ↓
Tokenization
      ↓
Lowercasing
      ↓
Word2Vec Training
      ↓
Word Embeddings
      ↓
Similarity Analysis
      ↓
Visualization
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* NLTK
* Gensim
* Matplotlib
* Scikit-learn
* Word2Vec
* Jupyter Notebook / Google Colab

## 🔍 Key Steps

### 1. Text Preprocessing

The raw book text is cleaned and converted into a format suitable for NLP processing.

Typical preprocessing includes:

* Removing unnecessary characters
* Converting text to lowercase
* Splitting text into sentences
* Tokenizing sentences into words
* Removing unwanted tokens

### 2. Tokenization

The text is converted into a list of sentences containing individual tokens.

Example:

```python
[
    ["jon", "snow", "went", "north"],
    ["arya", "was", "looking", "for", "jon"]
]
```

### 3. Training Word2Vec

The processed sentences are used to train the Word2Vec model.

Example:

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
```

### 4. Finding Similar Words

After training, we can find words that are semantically/contextually similar.

```python
model.wv.most_similar("jon")
```

Example output may look like:

```text
[('snow', 0.82),
 ('rob', 0.76),
 ('ned', 0.74),
 ...]
```

The exact results depend on preprocessing, corpus size, and model parameters.

### 5. Word Similarity

We can also calculate the similarity between two words:

```python
model.wv.similarity("king", "queen")
```

This returns a similarity score based on their learned vectors.

### 6. Word Embedding Visualization

Word embeddings can be reduced from high-dimensional space into 2D using techniques such as **PCA** or **t-SNE**.

This makes it possible to visually explore relationships between characters and concepts.

## 📊 Example Questions Explored

The trained model can be used to explore questions such as:

* Which words are similar to `jon`?
* Which characters appear in similar contexts?
* How closely related are `king` and `queen`?
* Which words are associated with `dragon`?
* What relationships can be discovered from the learned embeddings?

## 💡 What I Learned

Through this project, I learned:

* How NLP text preprocessing works
* How tokenization converts text into model-ready data
* How Word2Vec creates word embeddings
* The difference between CBOW and Skip-Gram
* How context helps models learn relationships between words
* How to calculate word similarity
* How to visualize high-dimensional embeddings
* How NLP models can discover semantic/contextual relationships from text

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/game-of-thrones-word2vec.git
```

### 2. Navigate to the project

```bash
cd game-of-thrones-word2vec
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
notebooks/game_of_thrones_word2vec.ipynb
```

You can also run the notebook directly in **Google Colab**.

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
nltk
gensim
matplotlib
scikit-learn
jupyter
```

## 📁 Project Structure

```text
game-of-thrones-word2vec/
│
├── data/
│   └── got_books.txt
│
├── notebooks/
│   └── game_of_thrones_word2vec.ipynb
│
├── images/
│   └── word2vec_visualization.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔮 Future Improvements

Some possible improvements include:

* Train the model on a larger corpus
* Experiment with different `vector_size` values
* Compare CBOW and Skip-Gram
* Tune `window` and `min_count`
* Use t-SNE for better embedding visualization
* Build a character recommendation system
* Create an interactive Streamlit application
* Compare Word2Vec with modern embeddings such as BERT

## 👨‍💻 Author

**Sagar Pravin Sapkale**

Engineering Student | AI & Data Science Enthusiast

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Data Science
* AI Engineering

---

⭐ If you found this project useful, consider giving the repository a star!

