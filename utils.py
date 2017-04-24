from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import chi2

from itertools import combinations
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from bokeh.io import output_notebook, show
from bokeh.charts import Scatter, Histogram

output_notebook()

plt.style.use('ggplot')

n_topics = 4
n_top_words = 25

def get_top_words(model, feature_names, n_top_words):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words.extend([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return top_words



def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #{}:".format(topic_idx + 1))
        print(" - ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
def label_data_topics(df, num_topics=4, text='Body', ngram=1):
    tf_vectorizer = CountVectorizer(max_df=0.80, min_df=50,
                                    max_features=None,
                                    stop_words='english',
                                    ngram_range=(1, ngram))

    tf = tf_vectorizer.fit_transform(df['Body'])

    #define the lda function, with desired options
    #Check the documentation, linked above, to look through the options
    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=20,
                                    learning_method='online',
                                    learning_offset=80.,
                                    total_samples=len(df['Body']),
                                    random_state=0)
    #fit the model
    data = lda.fit_transform(tf)
    labels = np.argmax(data, axis=1)
    df['Topic'] = labels
    return df, lda, tf_vectorizer

def significance_labeller(df, delta='Dow Jones Delta'):
    labels = []
    mean = df[delta].mean()
    std_dev = df[delta].std()
    for delta in df[delta]:
        if delta < mean - std_dev:
            labels.append(-1)
        elif mean - std_dev <= delta <= mean + std_dev:
            labels.append(0)
        else:
            labels.append(1)
    df['Label'] = labels
    return df


def featurize(df, k=100, text='Body', ngram=1):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, ngram))
    vectorizer.fit(df[text])
    words = vectorizer.get_feature_names()
    data = vectorizer.transform(df[text])
    labels = df['Label']
    
    print("There are %d datapoints that have a upward label" %(len(np.where(labels == 1)[0])))
    print("There are %d datapoints that have a neutral label" %(len(np.where(labels == 0)[0])))
    print("There are %d datapoints that have a downward label" %(len(np.where(labels == -1)[0])))

    chi_scores, p_vals = chi2(data, labels)
    
    word_scores = []
    for el in zip(chi_scores, words, p_vals):
        if not np.isnan(el[0]):
            word_scores.append(el)
            
    chi_scores, words, p_vals = zip(*word_scores)
    words = np.array(words)
    
    top_words_indices = (np.argsort(chi_scores)[::-1])[:k]
    top_words = words[top_words_indices]
    
    chi_vectorizer = CountVectorizer(vocabulary=top_words)
    data = chi_vectorizer.fit_transform(df[text])
    df = pd.DataFrame(np.hstack((data.todense(), labels.values.reshape((labels.shape[0], 1)))), columns=list(top_words) + ['label'])

    return df, chi_vectorizer

def plot_2d_scatter(df, vectorizer, text='Body', to_plot='Label'):
    X = vectorizer.transform(df[text])
    cos_dist = 1 - cosine_similarity(X.todense())
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    pos = mds.fit_transform(cos_dist)# shape (n_components, n_samples)
    distance_df = pd.DataFrame(pos, columns=['x', 'y'])
    distance_df[to_plot] = df[to_plot]
    p = Scatter(distance_df, x='x', y='y', title="MDS: White House Posts", color=to_plot,
           legend="top_right")

    show(p)

def join_blog_posts_on_date(df):
    posts = {}
    titles = {}
    for index, row in df.iterrows():
        date = row['Date']
        if date not in posts:
            posts[date] = []
            titles[date] = []
        posts[date].append(row['Body'])
        titles[date].append(row['Title'])

    posts = {date: ' '.join(posts[date]) for date in posts}
    titles = {date: ' '.join(titles[date]) for date in titles}

    posts = pd.DataFrame(list(posts.items()))
    posts.columns = ['Date', 'Body']

    titles = pd.DataFrame(list(titles.items()))
    titles.columns = ['Date', 'Title']

    dj_deltas = df[['Date', 'Dow Jones Delta']].drop_duplicates()
    nd_deltas = df[['Date', 'Nasdaq Delta']].drop_duplicates()
    sp_deltas = df[['Date', 'S&P 500 Delta']].drop_duplicates()

    dj_delta_prop = (df['Dow Jones Delta'] / df['Dow Jones Value']).drop_duplicates()
    nd_delta_prop = (df['Nasdaq Delta'] / df['Nasdaq Value']).drop_duplicates()
    sp_delta_prop = (df['S&P 500 Delta'] / df['S&P 500 Value']).drop_duplicates()

    dj_deltas['Dow Jones Proportion'] = dj_delta_prop
    nd_deltas['Nasdaq Proportion'] = nd_delta_prop
    sp_deltas['S&P 500 Proportion'] = sp_delta_prop

    dataset = pd.merge(posts, titles, how='inner', on=['Date'])
    dataset = pd.merge(dataset, dj_deltas, how='inner', on=['Date'])
    dataset = pd.merge(dataset, nd_deltas, how='inner', on=['Date'])
    dataset = pd.merge(dataset, sp_deltas, how='inner', on=['Date'])

    dataset['Body'] = dataset['Body'].str.replace('\d+', '').str.replace('[^a-zA-Z ]', '')
    dataset['Title'] = dataset['Title'].str.replace('\d+', '').str.replace('[^a-zA-Z ]', '')
    dataset['Mean Delta'] = (dataset['Dow Jones Delta'] + dataset['Nasdaq Delta'] + dataset['S&P 500 Delta']) / 3
    dataset['Mean Proportion'] = (dataset['Dow Jones Proportion'] +
                                  dataset['Nasdaq Proportion'] +
                                  dataset['S&P 500 Proportion']) / 3
    dataset['Label'] = (dataset['Mean Proportion'] >= 0).apply(int)
    dataset['Label'] = 2 * dataset['Label'] - 1

    return dataset