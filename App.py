import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
load_dotenv('.env')


def Authentication():
    consumer_key = os.environ["API_KEY"]
    consumer_secret = os.environ["API_KEY_SECRET"]
    access_token = os.environ["ACCESS_TOKEN"]
    access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]
    bearer_token = os.environ["BEARER_TOKEN"]
    auth = tweepy.OAuth1UserHandler(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )
    api = tweepy.API(auth)
    return api


def getTweets(query):
    tweets = tweepy.Cursor(api.search_tweets,
                           query, count=100, lang='en', tweet_mode='extended').items(1000)

    # texts = getText(tweets)   truncated=False
    return tweets


def sentiment(tweets):
    positiveList = []
    negativeList = []
    neutralList = []
    for item in tweets:
        item.score = []
        sentence = item.full_text
        score = sid.polarity_scores(sentence)['compound']
        item.score.append(score)
        if (score > 0.1):
            item.full_text = item.full_text
            positiveList.append(item)
        if (score < -0.1):
            item.full_text = item.full_text
            negativeList.append(item)
        if (score < 0.1 and score > -0.1):
            item.full_text = item.full_text
            neutralList.append(item)

    return positiveList, negativeList, neutralList


def removeDuplicate(tweets):
    uniqBox = []
    removedTweet = []
    removedList = []
    for item in tweets:
        text = item.full_text
        hashNumber = text.count("#")
        if text not in uniqBox:
            if 5 >= hashNumber:
                uniqBox.append(text)
                removedTweet.append(item)
        else:
            removedList.append(item)
        if 5 < hashNumber:
            removedList.append(item)
    return removedTweet, removedList


def saveToCsv(tweets):
    userNameList = []
    screenNameList = []
    tweetList = []
    NLTK = []
    for item in tweets:
        userNameList.append(item.user.name)
        screenNameList.append(item.user.screen_name)
        tweetList.append(item.full_text)
        # -----------------------------------------------------
        sentence = item.full_text
        score = sid.polarity_scores(sentence)['compound']
        if (score > 0.1):
            NLTK.append("POSITIVE " + str(score))
        if (score < -0.1):
            NLTK.append("NEGATIVE " + str(score))
        if (score < 0.1 and score > -0.1):
            NLTK.append("NEUTRAL " + str(score))
        # -----------------------------------------------------
    df = pd.DataFrame({
        'User name': userNameList,
        'Handle': screenNameList,
        'Tweets': tweetList,
        'NLTK': NLTK
    })
    df.to_csv("./output.csv")


api = Authentication()

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


def getScore(x):
    return x.get('score')


@app.route("/searchTopic", methods=["POST", "GET"])
def searchTopic():
    name = ''
    output = request.form.to_dict()
    name = output["name"]
    print(output)
    if name == '':
        error = 'please put keyword'
        return render_template("index.html", error=error)
    tweet = getTweets(name)
    tweetList = list(tweet)
    removedTweet, removedList = removeDuplicate(tweetList)
    if len(removedTweet) > 100:
        firstHudredTweets = removedTweet[0:100]
    else:
        firstHudredTweets = removedTweet
    saveToCsv(firstHudredTweets)
    positiveList, negativeList, neutralList = sentiment(firstHudredTweets)
    positiveList = sorted(positiveList, key=lambda x: x.score, reverse=True)
    neutralList = sorted(neutralList, key=lambda x: x.score)
    negativeList = sorted(negativeList, key=lambda x: x.score)
    summary = {'query': name,
               'total_tweets': len(tweetList),
               'removed_list': len(removedList),
               'clear_tweets': len(removedTweet),
               'sentiment_input': len(firstHudredTweets),
               'positive': len(positiveList),
               'negative': len(negativeList),
               'neutral': len(neutralList)}
    return render_template("index.html", positive=positiveList, negative=negativeList, neutral=neutralList, removed=removedList, summary=summary)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
