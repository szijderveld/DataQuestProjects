
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

print(train.info())
target = 'SalePrice'

# import matplotlib.pyplot as plt
# # For prettier plots.
# import seaborn
# fig = plt.figure(figsize=(7,15))

# ax1 = fig.add_subplot(3, 1, 1)
# ax2 = fig.add_subplot(3, 1, 2)
# ax3 = fig.add_subplot(3, 1, 3)

# train.plot(x="Garage Area", y="SalePrice", ax=ax1, kind="scatter")
# train.plot(x="Gr Liv Area", y="SalePrice", ax=ax2, kind="scatter")
# train.plot(x="Overall Cond", y="SalePrice", ax=ax3, kind="scatter")

# plt.show()

#'Gr Liv Area' correlates with 'SalePrice' seen from plot
print(train[['Garage Area', 'Gr Liv Area', 'Overall Cond', 'SalePrice']].corr())


#univarient linear regression model
import numpy as np
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
from sklearn.metrics import mean_squared_error

train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])

train_mse = mean_squared_error(train_predictions, train['SalePrice'])
test_mse = mean_squared_error(test_predictions, test['SalePrice'])

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(train_rmse)
print(test_rmse)






#multiple linear regression
cols = ['Overall Cond', 'Gr Liv Area']
lr.fit(train[cols], train['SalePrice'])
train_predictions = lr.predict(train[cols])
test_predictions = lr.predict(test[cols])

train_rmse_2 = np.sqrt(mean_squared_error(train_predictions, train['SalePrice']))
test_rmse_2 = np.sqrt(mean_squared_error(test_predictions, test['SalePrice']))

print(train_rmse_2)
print(test_rmse_2)