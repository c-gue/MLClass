''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

from string import digits
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


#how many samples and How many features?
diabetes = datasets.load_diabetes()
print(diabetes.data.shape)
#422 samples
#10 features

# What does feature s6 represent?
print(diabetes.DESCR)
#s6 = glu, blood sugar level

#print out the coefficient
data_train, data_test, target_train, target_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

lr = LinearRegression()
lr.fit(X=data_train,y=target_train)

print(lr.coef_)


#print out the intercept
print(lr.intercept_)


# create a scatterplot with regression line
predicted = lr.predict(data_test)
expected = target_test

predict = (lambda x: lr.coef_*x + lr.intercept_)

import seaborn as sns

axes = sns.scatterplot(
    data = diabetes,
    x=predicted,
    y=expected,
    hue = predicted,
    #palette = "winter",
    legend = False
)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(diabetes.predicted.values),max(diabetes.predicted.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)
plt.show()
