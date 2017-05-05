#!/usr/bin/env python3

from datetime import date
import numpy as np
import pandas as pd
import requests
import bs4
from yahoo_finance import Share


INDEX_NAMES = {'Nasdaq': '^IXIC', 'Dow Jones': '^DJI', 'S&P 500': '^GSPC'}

def generate_wh_data(n, president='trump'):
    """
    Pulls White House Posts and writes them to
    data/WH_posts.csv
    """
    # new list to append urls
    post_urls = []
    if president == 'obama':
        url = "https://www.obamawhitehouse.archives.gov"
        s = 1
    else:
        url = "https://www.whitehouse.gov"
        s = 0
        
    print("Pulling White House posts from " + url)
    for i in np.arange(s, n):
        #grab url and append page number
        page_url = url + "/blog?page=" + str(i)
        try:
            r = requests.get(page_url)
        except requests.exceptions.ConnectionError:
            print('There was a connection error')
            break

        soup = bs4.BeautifulSoup(r.content.decode('utf-8'), "html.parser")
        # h3 field-content is the tag to get post urls
        page_posts = soup.find_all("h3", "field-content")
        # dates for the posts
        post_dates = soup.find_all("span", "field-content")
        # append page's post urls to our list
        post_urls.append([(post.text, post.find('a')['href'], date.text) for post, date in zip(page_posts, post_dates)])

    # flatten list
    all_posts = [p for post in post_urls for p in post]

    posts = []

    # loop through every post and get text in post
    print("Parsing each post")
    for post_data in all_posts:
        req = requests.get(url + post_data[1], allow_redirects=False)
        if req.status_code != 200:
            all_posts.remove(post_data)
        else:
            req_soup = bs4.BeautifulSoup(req.text, "html.parser")
            post_body = req_soup.find("div","pane-entity-field").text\
            .encode('ascii', 'ignore').decode('UTF-8').replace("\n", "").replace("\t", "")
            posts.append(post_body)

    df_out = pd.DataFrame({'a': [post[2].strip() for post in all_posts], 'b': [post[0] for post in all_posts], 'c': [post.strip() for post in posts]})
    df_out.columns = ['Date', 'Title', 'Body']

    print("Writing data/WH_posts.csv")
    df_out.to_csv('data/WH_posts.csv', index=False)
    return df_out

def get_stock_values(stock_abbrv):
    """
    Given the stock abbrevation, this function
    will pull from Yahoo Finance the history of that
    stock from 2016-12-25 to present day
    """
    share = Share(stock_abbrv)
    share_history = share.get_historical('2016-12-25', date.isoformat(date.today()))
    df = pd.DataFrame([[s['Date'], float(s['Close'])] for s in share_history], columns=['Date', 'Value'])
    return df

def create_dataset(regenerate=False, n=50, president='trump'):
    """
    Pulls closing values from Yahoo finance and
    matches those values to white house posts using
    the date field. Returns a dataframe for easy
    manipulation.
    """

    if regenerate:
        wh_df = generate_wh_data(n, president)
    else:
        wh_df = pd.read_csv('data/WH_posts.csv')
    wh_df['Date'] = pd.to_datetime(wh_df['Date'])

    stock_dfs = {name:get_stock_values(INDEX_NAMES[name]) for name in INDEX_NAMES}
    processed_stock_dfs = []
    for name in stock_dfs:
        stock_df = stock_dfs[name]
        dates = pd.to_datetime(stock_df['Date'])
        values = stock_df['Value'].rename(name + ' Value')
        stock_df = pd.concat([dates, values], axis=1, join_axes=[dates.index])
        stock_df.sort_values(by='Date')

        stock_df[name + ' Delta'] = -stock_df[name + ' Value'].diff(periods=1)
        stock_df[name + ' Proportion'] = stock_df[name + ' Delta'] / stock_df[name + ' Value']
        processed_stock_dfs.append(stock_df)


    wh_df['Date'] = pd.to_datetime(wh_df['Date'])
    stock_df = processed_stock_dfs[0]
    for i in range(1, len(stock_dfs)):
        stock_df = pd.merge(stock_df, processed_stock_dfs[i], how='inner', on=['Date'])

    dataset = pd.merge(wh_df, stock_df, how='inner', on=['Date'])
    dataset.to_csv('data/dataset-' + president + '.csv', index=False)

    return dataset.sort_values(by='Date').reset_index(drop=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='White House Post and Stock Market Collector')
    parser.add_argument('-r', '--regen', action='store_true',
                    help='Regenerates WH_post.csv')
    parser.add_argument('-n', '--num_pages', type=int,
                    help='Number of pages to pull')
    parser.add_argument('-p', '--president', type=str,
                       help='trump or obama')
    args = parser.parse_args()

    create_dataset(regenerate=args.regen, n=args.num_pages, president=args.president)
