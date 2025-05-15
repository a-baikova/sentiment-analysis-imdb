# 🎬 IMDb Reviews Sentiment Analysis

## Project Overview
This group project uses This group project uses the IMDb Large Movie Review Dataset of 43,750 labeled reviews to build a sentiment analysis model that classifies movie reviews as positive or negative. 
We systematically compared baseline tools, lexicon-based classifiers, classic machine learning models, transformer-based architectures, and generative LLMs for sentiment classification. 

## Methodology
1. **Data**:
- IMDb: 43,750 balanced reviews (train/test)
- Portuguese laptop reviews: 1,500 labeled and 421 unlabeled

2. **Baselines & Lexicons**
- Flair (90.3%), Bing Liu (72.9%), others (61–65%)

![image](https://github.com/user-attachments/assets/4df2edc8-be07-45d5-90be-4a273a311813)


3. **Machine Learning**
- Preprocessing: Various methods were tested such as HTML/URL removal, case folding, tokenization, stopword removal, stemming/lemmatization, POS tagging, etc.
- Feature Generation: TF-IDF, BoW, Word2Vec, FastText
- Models: Logistic Regression, SVM, Multinomial Naive Bayes, MLP

![image](https://github.com/user-attachments/assets/c2effedd-65b9-4b3e-9b7a-75d4f5b5258b)
![image](https://github.com/user-attachments/assets/81e23437-822c-4dd4-a42f-2ad561b6f71f)

2. **Transformers**
- Prepocessing: Only special characters were removed.
- Models: DistilBERT and TextAttack Bert Uncased IMDb without finetuning were tested, and DistilBERT base and RoBERTa base were tested with finetuning

![image](https://github.com/user-attachments/assets/6f2072ec-cb97-4923-888e-a3a6ba3a1b9a)

3. **LLMs (Portuguese dataset)**

- Models: We tested Gervásio, GlorIA, with prompt engineering, and Granite and Mistral with simple prompts

![image](https://github.com/user-attachments/assets/e77d6021-2756-4993-8dee-84ad54a25f41)

4. **Tools & Libraries**
Python, Jupyter Notebook, scikit-learn, NLTK, HuggingFace, Transformers, Flair, Matplotlib, Seaborn

5. **Results**
- Classic ML: TF-IDF + Logistic Regression is robust and efficient with 89,9% accuracy
- Transformers: RoBERTa achieves the highest accuracy, but with some overfitting
- LLMs: Gervásio is best for Portuguese, with optimized prompts
