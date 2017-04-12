import pandas as pd


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