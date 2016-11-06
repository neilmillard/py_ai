import os
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# plt.switch_backend('new_backend')
dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        # load file into variable
        csv_file_reader = csv.reader(csvfile)
        # use next to skip the first row as it contains header labels
        next(csv_file_reader)
        for row in csv_file_reader:
            # each row get the date [0] as an int
            dates.append(int(row[0].split('-')[0]))
            # append the open price as a float value
            prices.append(float(row[1]))
    return


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


try:
    os.chdir('data')
    get_data('aapl.csv')
except OSError as err:
    print("os error: {0}".format(err))
    cwd = os.getcwd()
    print("cwd: {0}".format(cwd))
    for entry in os.scandir('.'):
        if entry.is_file():
            print(entry.name)
    exit()
predicted_price = predict_prices(dates, prices, 29)
print(predicted_price)
