import pandas as pd
import quandl, math,datetime,time
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = pd.read_csv('quandl.csv').set_index('Date')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close']*100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open']*100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:] 
X = X[:-forecast_out]



df.dropna(inplace = True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size= 0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

with open ('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)
forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan
 
last_date = df.iloc[-1].name
dtime = pd.date_range(last_date, periods=forecast_out+1, freq = 'D')
index= 1


for i in forecast_set:
    df.loc[dtime[index]] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    index += 1


df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


