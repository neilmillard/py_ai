# https://github.com/ciurana2016/predict_stock_py
# # Dependencies
# * numpy (http://www.numpy.org/)
# * tweepy (http://www.tweepy.org/)
# * textblob (https://textblob.readthedocs.io/en/dev/)
# * requests(http://docs.python-requests.org/en/master/)
# * keras(https://keras.io/) Runs with [TensorFlow](https://www.tensorflow.org/)
#                           or [Theano](http://deeplearning.net/software/theano/), so you will need one of them.
#
import os
import sys
import tweepy
import requests
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from textblob import TextBlob

# First we login into twitter
consumer_key = 'hxYAbUapce865P5xF7TkE5qwG'
consumer_secret = 'jNoLT0RjrmBhnIa9OtHouiBwahnnz63eLgubh7gTKt6ImMFOCA'
access_token = '745177828212105216-I4FM0l27G2042mx3C7KC1ImBw3xUw0N'
access_token_secret = 'WtwHFaRbLNAkLcEN9xpeAeBwIffQCR1li0L2kyclmLp95'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

# Where the csv file will live
FILE_NAME = 'historical.csv'


def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1

    if positive > ((num_tweets - null) / 2):
        return True


def get_historical(quote):
    # Download our file from google finance
    url = 'http://www.google.com/finance/historical?q=NASDAQ%3A' + quote + '&output=csv'
    r = requests.get(url, stream=True)

    if r.status_code != 400:
        with open(FILE_NAME, 'wb') as f:
            for chunk in r:
                f.write(chunk)

        return True


def stock_prediction():
    # Collect data points from csv
    dataset = []

    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[1]))

    dataset = np.array(dataset)

    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n + 1] for n in range(len(dataset) - 2)]
        return np.array(dataX), dataset[2:]

    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    result = 'The price will move from %s to %s' % (dataset[0], prediction[0][0])

    return result


# Ask user for a stock quote
# raw_input('Enter a stock quote from NASDAQ (e.j: AAPL, FB, GOOGL): ').upper()
stock_quote = 'AAPL'

# Check if the stock sentiment is positve
if not stock_sentiment(stock_quote, num_tweets=100):
    print
    'This stock has bad sentiment, please re-run the script'
    sys.exit()

# Check if we got te historical data
if not get_historical(stock_quote):
    print
    'Google returned a 404, please re-run the script and'
    print
    'enter a valid stock quote from NASDAQ'
    sys.exit()

# We have our file so we create the neural net and get the prediction
print
stock_prediction()

# We are done so we delete the csv file
os.remove(FILE_NAME)