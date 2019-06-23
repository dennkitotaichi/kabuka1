
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# データ読み込み
df = pd.read_csv("stock_nikkei_180321.csv")
df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')

#データの標準化
df['close'] = preprocessing.scale(df['close'])
df = df.loc[:, ['date', 'close']]

#訓練、テストデータの作成
maxlen = 10
X, Y = [], []
for i in range(len(df) - maxlen):
    X.append(df[['close']].iloc[i:(i+maxlen)].as_matrix())
    Y.append(df[['close']].iloc[i+maxlen].as_matrix())
X=np.array(X)
Y=np.array(Y)

# 訓練用のデータと、テスト用のデータに分ける
N_train = int(len(df) * 0.8)
N_test = len(df) - N_train
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=N_test, shuffle = False) 

# 隠れ層の数などを定義: 隠れ層の数が大きいほど精度が上がる?
n_in = 1 # len(X[0][0])
n_out = 1 # len(Y[0])
n_hidden = 300

#モデル作成 (Kerasのフレームワークで簡易に記載できる)
model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, maxlen, n_in),
               kernel_initializer='random_uniform',
               return_sequences=False))
model.add(Dense(n_in, kernel_initializer='random_uniform'))
model.add(Activation("linear"))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss = "mean_squared_error", optimizer=opt)

early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
hist = model.fit(X_train, y_train, batch_size=maxlen, epochs=50,
                 callbacks=[early_stopping])

# 損失のグラフ化
loss = hist.history['loss']
epochs = len(loss)
plt.rc('font', family='serif')
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.show()

# 予測結果
predicted = model.predict(X_test)
result = pd.DataFrame(predicted)
result.columns = ['predict']
result['actual'] = y_test
result.plot()
plt.show()