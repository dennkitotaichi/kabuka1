def gen_train(self, seq_len):
    """
    Generates training data
    :param seq_len: length of window
    :return: X_train and Y_train
    """
    for i in range((len(self.stock_train)//seq_len)*seq_len - seq_len - 1):
        x = np.array(self.stock_train.iloc[i: i + seq_len, 1])
        y = np.array([self.stock_train.iloc[i + seq_len + 1, 1]], np.float64)
        self.input_train.append(x)
        self.output_train.append(y)
    self.X_train = np.array(self.input_train)
    self.Y_train = np.array(self.output_train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, Y_train, epochs=100)

model.evaluate(X_test, Y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, Y_train, epochs=50)

model.evaluate(X_test, Y_test)

def back_test(strategy, seq_len, ticker, start_date, end_date, dim):
    """
    A simple back test for a given date period
    :param strategy: the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param seq_len: length of the days used for prediction
    :param ticker: company ticker
    :param start_date: starting date
    :type start_date: "YYYY-mm-dd"
    :param end_date: ending date
    :type end_date: "YYYY-mm-dd"
    :param dim: dimension required for strategy: 3dim for LSTM and 2dim for MLP
    :type dim: tuple
    :return: Percentage errors array that gives the errors for every test in the given date range
    """
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    stock_data = data["Adj Close"]
    errors = []
    for i in range((len(stock_data)//10)*10 - seq_len - 1):
        x = np.array(stock_data.iloc[i: i + seq_len, 1]).reshape(dim) / 200
        y = np.array(stock_data.iloc[i + seq_len + 1, 1]) / 200
        predict = strategy.predict(x)
        while predict == 0:
            predict = strategy.predict(x)
        error = (predict - y) / 100
        errors.append(error)
        total_error = np.array(errors)
    print(f"Average error = {total_error.mean()}")


