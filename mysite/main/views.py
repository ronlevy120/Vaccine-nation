from django.shortcuts import render
import numpy as np
from .forms import CreateNewList
import pickle
import pandas as pd
pd.set_option('display.max_colwidth', None)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re

model = pickle.load(open('mysite/main/xgboost.pkl', 'rb'))

def home(response):
    return render(response, 'main/home.html', {})

def idea(response):
    return render(response, "main/idea.html", {})


def team(response):
    return render(response, "main/team.html", {})

mostcommon = ['vaccine', 'get', 'covid', 'make', 'today', '19', 'vaccinate', 'antivaxx', 'anti', 'first', 'time',
              'people',
              'vaccination', 'receive', 'thank', 'dose', 'week', '2', '1', 'help', 'day', 'one', 'staff', 'home',
              'take', 'health', 'safety',
              'new', 'interest', 'see', 'theorist', 'u', 'efficacy', 'raise', 'conflict', 'query', 'work',
              'youconspiracy', 'shot',
              'community', 'second', 'continue', 'go', 'say', 'care', 'resident', 'prediction', 'need', '2nd', 'every',
              'safe', 'year',
              'good', 'administer', 'know', 'state', 'protect', 'give', 'worker', 'dr', 'visit', 'share', 'family',
              '11', 'million',
              'team', 'part', 'impact', 'come', 'important', 'nothing', 'watch', 'keep', 'across', 'kentucky', 'many',
              'much', '000',
              'thanks', '37', '6', '22', 'question', 'waste', 'great', 'change', 'life', 'plus', 'footcouple',
              'notdrop', 'like', 'want',
              'use', 'love', 'well', 'old', 'moment', 'last', 'look', 'roll']

contractions_dict = {"ain't": "are not ", "'s": " is ", "aren't": "are not",
                     "can't": "cannot", "can't've": "cannot have",
                     "'cause": "because", "could've": "could have", "couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                     "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                     "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                     "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                     "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
                     "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                     "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                     "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                     "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have", "they'll": "they will",
                     "they'll've": "they will have", "they're": "they are", "they've": "they have",
                     "to've": "to have", "wasn't": "was not", "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                     "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                     "what'll've": "what will have", "what're": "what are", "what've": "what have",
                     "when've": "when have", "where'd": "where did", "where've": "where have",
                     "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                     "why've": "why have", "will've": "will have", "won't": "will not",
                     "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                     "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have", "y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                     "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def most_common_token(token, mostcommon=mostcommon):
    new_token = []
    for word in token:
        if word in mostcommon:
            new_token.append(word)
    return new_token


def preprocess_input(user_input, stop_wrds=set(stopwords.words('english')), wl=WordNetLemmatizer()):
    x = user_input
    for_df = {'text': [x], 'user_followers': [2207], 'favorites': [0], 'retweets': [0], 'is_retweet': [1]}
    df = pd.DataFrame(for_df)

    df.text = df.text.str.lower()
    df.text = df.text.apply(lambda x: re.sub('@[^\s]+', '', x))
    df.text = df.text.apply(lambda x: re.sub(r'\B#\S+', '', x))
    df.text = df.text.apply(lambda x: re.sub(r"http\S+", "", x))
    df.text = df.text.apply(lambda x: ' '.join(re.findall(r'\w+', x)))
    df.text = df.text.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))
    df.text = df.text.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

    df['text'] = df['text'].apply(lambda x: expand_contractions(x))
    df['tokenized'] = df['text'].apply(word_tokenize)
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
    df['pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)
    df['tokenized'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['tokenized'] = df['tokenized'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    df['n_words'] = df['text'].apply(lambda x: len(x.split()))
    df['tokenized_common'] = df['tokenized'].apply(lambda x: most_common_token(x))

    for word in mostcommon:
        df[word] = 0
    for word in df.tokenized_common:
        df[word] = 1

    to_drop = ['text', 'tokenized', 'pos_tags', 'tokenized_common']
    df = df.drop(to_drop, axis=1)

    return df

def test(response):
    if response.method == "POST":
        form = CreateNewList(response.POST)
        print(f"form: {form}")
        if form.is_valid():
            print("FORM IS VALID")
            n = form.cleaned_data["name"]
            print(f"n: {n}")
            result = preprocess_input('hhh')
            print(f"result: {result}")
            # df = preprocess_input(n)
            # print(f"df: {df}")
            # preds = model.predict_proba(df)
            # result = np.asarray([np.argmax(line) for line in preds])[0]
            print(f"result type: {type(result)}, result: {result}")
        else:
            print("NOT VALID")
            result = "NOT VALID"
        return render(response, 'main/test.html', {"form": form, "output": result})

    else:
        form = CreateNewList()
    return render(response, 'main/test.html', {"form": form})

