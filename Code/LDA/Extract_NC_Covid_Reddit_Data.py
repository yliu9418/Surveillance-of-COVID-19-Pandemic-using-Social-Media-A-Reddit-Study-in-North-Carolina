
# coding: utf-8

import praw
import pandas as pd
import datetime as dt
import time
import os
from psaw import PushshiftAPI



reddit = praw.Reddit(client_id= 'put_yours_here', client_secret='put_yours_here',
                     password='put_yours_here', user_agent='put_yours_here',
                     username='put_yours_here')

api = PushshiftAPI(reddit)
query = 'corona virus | Coronavirus | COVID-19 | SARS-CoV-2'
start_epoch=int(dt.datetime(2020, 3, 1).timestamp())
end_epoch = int(dt.datetime(2020, 8, 31).timestamp())
subred_list = ['Charlotte','gso', 'NorthCarolina','raleigh']

def getData(sName):
    results = api.search_comments( q=query, after=start_epoch, before=end_epoch, subreddit=sName)
    cache = []
    ID = []
    for c in results:
        if c.submission not in ID:
            ID.append(c.submission)
        cache.append(c)
    return cache

from praw.models import MoreComments

def convert_date(created):
    date_time = dt.datetime.fromtimestamp(created)
    return date_time.strftime("%m/%d/%Y")

def saveResults(submission_list,sName):
    commentData = []
    for comment in submission_list:
        comments = []
        sub = reddit.submission(id = comment.submission)
        sub.comments.replace_more(None)
        for comment in sub.comments.list():
            comments.append(comment.body)
        commentData.append([comment.submission.created_utc, comment.submission.title, comment.body ,comment.submission.num_comments ])
    commentData = pd.DataFrame(commentData,columns=['Date','submission title', 'comments', '# of comments'])
    commentData['Date'] = commentData['Date'].apply(convert_date) 
    commentData.sort_values('Date', inplace = True, ascending = False)
    commentData.to_csv(os.getcwd() + r'/reddit_data/%s.csv' %sName, index=False, encoding="utf-8")

for i in subred_list:
    saveResults(getData(i),i)

