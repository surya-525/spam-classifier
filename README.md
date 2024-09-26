# Spam-Ham-Classifier 
## A Machine learning classifier to predict whether the SMS is Spam or Ham by using Natural Language Processing (NLP)

<h1 align="center">
  <br>
  <a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/logo1.png" alt="SpamClassifier"></a>
  <br>
  SpamClassifier
  <br>
</h1>
<h4 align="center">In this project I build a model for classifying the SMS/Email into spam or ham through the text of the SMS/Email using standard classifiers.</h4>


## Table of Content

  * [Dataset](#dataset)
  * [Demo](#demo)
  * [Screenshots](#screenshots)
  * [Methodology](#methodology)
  * [Bug / Feature Request](#bug--feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Contact](#contact) 
  
## Dataset
The SMS/Email Spam Collection is a set of SMS tagged messages that have been collected for SMS/Email Spam research. It contains one set of SMS messages in English of 5,567 messages, tagged according being ham (legitimate) or spam.

> You can collect raw dataset from [here](https://github.com/Pratik180198/Spam-Ham/blob/master/spam.csv).

The files contain one message per line. Each line is composed by two columns:
- `Class`- contains the label (ham or spam) 
- `Message` - contains the raw text.

Dataset link : https://www.kaggle.com/uciml/sms-spam-collection-dataset

## Demo
Link: [https://spam-ham-nlp-model.herokuapp.com/](https://spam-ham-nlp-model.herokuapp.com/)

## Screenshots

#### CHECKING PART 1 

<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(69).png"></a>
<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(70).png"></a>

#### CHECKING PART 2

<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(72).png"></a>
<a href="url"><img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/Screenshot%20(71).png"></a>

## Methodology

### 1. Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

### 2. Create Virtual Environment
It is always a good practise to create a virtual environment and mostly it is very useful while deploying app. To create virtual environment for python and jupyter follow this link : https://janakiev.com/blog/jupyter-virtual-envs/

### 3. (Natural Language Toolkit) NLTK:
NLTK is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks.
Installing NLTK Library
```bash
!pip install nltk 
```
Type above code in the Jupyter Notebook or if it doesn’t work, type this in your cmd prompt "pip install nltk". This should work in most cases. Install NLTK: http://pypi.python.org/pypi/nltk

Importing NLTK Library

-> import nltk

-> nltk.download()

Download the packages.

### 4. Reading and Exploring Dataset

#### Reading in text data & why do we need to clean the text?
While reading data, we get data in the structured or unstructured format. A structured format has a well-defined pattern whereas unstructured data has no proper structure. In between the 2 structures, we have a semi-structured format which is a comparably better structured than unstructured format. When we read semi-structured data it is hard to interpret so we use pandas to easily understand our data.

### 5. Pre-processing Data
Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:

#### i. Remove punctuation
Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you

#### ii.Tokenization
Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.

#### iii. Remove stopwords
Stopwords are common words that will likely appear in any text. They don’t tell us much about our data so we remove them. eg: silver or lead is fine for me-> silver, lead, fine.

### 6. Preprocessing Data: Stemming
Stemming helps reduce a word to its stem form. It often makes sense to treat related words in the same way. It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl Note: Some search engines treat words with the same stem as synonyms.

### 7. Preprocessing Data: Lemmatizing
Lemmatizing derives the canonical form (‘lemma’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle In Short, Stemming is typically faster as it simply chops off the end of the word, without understanding the context of the word. Lemmatizing is slower and more accurate as it takes an informed analysis with the context of the word in mind.

### 8. Vectorizing Data
Vectorizing is the process of encoding text as integers i.e. numeric form to create feature vectors so that machine learning algorithms can understand our data.

#### i. Vectorizing Data: Bag-Of-Words
Bag of Words (BoW) or CountVectorizer describes the presence of words within the text data. It gives a result of 1 if present in the sentence and 0 if not present. It, therefore, creates a bag of words with a document-matrix count in each text document.

#### ii. Vectorizing Data: N-Grams
N-grams are simply all combinations of adjacent words or letters of length n that we can find in our source text. Ngrams with n=1 are called unigrams. Similarly, bigrams (n=2), trigrams (n=3) and so on can also be used.

Unigrams usually don’t contain much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the letter or word is likely to follow the given word. The longer the n-gram (higher n), the more context you have to work with.

#### iii. Vectorizing Data: TF-IDF
It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents). Note: Used for search engine scoring, text summarization, document clustering.

TF-IDF is applied on the body_text, so the relative count of each word in the sentences is stored in the document matrix. (Check the repo). Note: Vectorizers outputs sparse matrices. Sparse Matrix is a matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the locations of the non-zero elements.

<p align="center">
  <br>
  <img src="https://github.com/Pratik180198/Spam-Ham/blob/master/Screenshots/modelLearning.png">
</p>


### 9. Making Model and Evaluation Metrics

In this Spam-Ham NLP project we had used Support Vector Classifier, Logistic Regression, Naive Bayes, Random Forest Classifier in which SVC works well with 99.83% accuracy. Evaluation metrics are also used like accuracy_score, confusion metrics, classification report.

### 10. Model Deployment

This web app is made with the help of Flask. The Web UI is made by Anuj Vyas thanks for the UI [Anuj Vyas](https://github.com/anujvyas).

To deploy this app we are using Heroku Platform. You must first register on [Heroku](https://www.heroku.com/home).
Create your new app and give the app name and start deploying with the help of Heroku CLI command or connect your Github account.
After successful connecting search your application repository and then start your deploying process.
Once the app is successfully build you can visit your web app.

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

To run the app, create a procfile and shoot this command in the project directory:
```bash
web: gunicorn app:app
```


## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/Pratik180198/Spam-Ham/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/Pratik180198/Spam-Ham/issues/new). Please include sample queries and their corresponding results.

## Technologies Used

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/><img alt="Flask" src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white"/><img alt="PyCharm" src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green"/><img alt="GitHub" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"/><img alt="Heroku" src="https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white"/><img alt="Pandas" src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" /><img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" /><img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /><img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />

## Team

[<img target="_blank" src="https://avatars.githubusercontent.com/u/72552513?v=4" width=170>](https://github.com/Pratik180198) |
-|
[Pratik Bambulkar](https://github.com/Pratik180198) |)


## Contact

You can reach me : 

[<img alt="Instagram" src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>](https://www.instagram.com/pratikkk______/)
[<img alt="Gmail" src= "https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>](https://mail.google.com/mail/?view=cm&fs=1&to=pratikbambulkar1818@gmail.com&su=Bank_Note_Authentication&body=BODY)
[<img alt="Facebook" src= "https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white"/>](http://www.facebook.com/100004659334096/)
[<img alt="Linkedin" src= "https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/pratik-bambulkar-06241116a/)
