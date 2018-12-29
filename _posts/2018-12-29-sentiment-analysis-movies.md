---
layout: post
title: "Sentiments Analysis of Movie Dialogs"
date: 2018-12-29
---
In this task I get a movie dialogs data, and do *Sentiment Analysis* on it using **TextBlob** and **NLTK** libraries.

## Movie Dialog Dataset
I am going to use this movie dialog corpus on Kaggle as my dataset of movie dialogs:  
https://www.kaggle.com/Cornell-University/movie-dialog-corpus#movie_lines.tsv

This is a tab separated file of dialogs. Each line is of following format:
```
L1045	u0	m0	BIANCA	They do not!
```

I am going to use pandas to load this file.  
Note:
1. File has no header
2. Seperator is tab
3. I only want the last column (4)
4. And I want a list of it


```python
import pandas as pd
```


```python
dialog_list = pd.read_csv('movie_lines.tsv', header=None, usecols=[4], sep='\t')[4].tolist()
```

## Explore Sentiment Analysis Libraries
One kind of analysis I am going to do is related to sentiment polarity. I.e. to explore the distribution of positive, negative and neutral sentiments in movie dialogs. Several NLP libraries can be used for this. I am going to begin with the following two:
  * TextBlob: https://textblob.readthedocs.io/en/dev/
  * NLTK: https://www.nltk.org/
Let me now explore them one by one

### TextBlob


```python
from textblob import TextBlob
```


```python
sentiment_scores_tb = [round(TextBlob(str(dialog)).sentiment.polarity, 2)
                       for dialog in dialog_list]
sentiment_category_tb = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' 
                         for score in sentiment_scores_tb]
```

let's plot and see now


```python
import seaborn as sns
```


```python
# sentiment statistics
df_pd = pd.DataFrame([sentiment_category_tb]).T
df_pd.columns = ['Sentiment Category']

# Draw a catplot to show count by sentiment_category
g_pd = sns.catplot(x="Sentiment Category", data=df_pd, kind="count")
g_pd.set_ylabels("Dialog Count")
```


![png](/images/sentiment-analysis-movies_files/sentiment-analysis-movies_10_0.png)


### NLTK
I am going to use NLTK’s built-in Vader Sentiment Analyzer. This will simply rank a piece of text as positive, negative or neutral using a lexicon of positive and negative words.


```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
```


```python
sia = SentimentIntensityAnalyzer()
sentiment_scores_nltk = [sia.polarity_scores(str(dialog)) for dialog in dialog_list]
```


```python
sentiment_category_nltk = ['positive' if score['compound'] > 0.25
                           else 'negative' if score['compound'] < -0.25
                           else 'neutral'
                           for score in sentiment_scores_nltk]
```


```python
# Without placing into specifica categories (neutral, positive, negative)
# I can just sum up the category distributions
df_nltk_sum = pd.DataFrame(pd.DataFrame(sentiment_scores_nltk).sum())
df_nltk_sum.columns = ['Sum']
df_nltk_sum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>compound</th>
      <td>11075.3443</td>
    </tr>
    <tr>
      <th>neg</th>
      <td>24022.3960</td>
    </tr>
    <tr>
      <th>neu</th>
      <td>242111.7730</td>
    </tr>
    <tr>
      <th>pos</th>
      <td>35821.8470</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sentiment statistics
df_nltk = pd.DataFrame([sentiment_category_nltk]).T
df_nltk.columns = ['Sentiment Category']

# Draw a catplot to show count by sentiment_category
g_nltk = sns.catplot(x="Sentiment Category", data=df_nltk, kind="count")
g_nltk.set_ylabels("Dialog Count")
```


![png](/images/sentiment-analysis-movies_files/sentiment-analysis-movies_16_0.png)

