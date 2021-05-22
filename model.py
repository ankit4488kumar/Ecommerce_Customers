# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pickle
#Read csv file
df = pd.read_csv('Ecommerce_Customers.csv')
#Head of dataframe
df.head()
#Analysis of Dataframe
df.describe()
df.info()
#visualization of dataframe
sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = df)
plt.savefig('Time_on_websiteVsYearly_amount_spent.png')
sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = df)
plt.savefig('Time_on_AppVsYearly_amount_spent.png')

sns.jointplot(x='Time on App',y ='Length of Membership', data = df, kind='hex')
plt.savefig('Time_on_websiteVsLength_of_Membership.png')
sns.pairplot(df)
plt.savefig('pairplot.png')
sns.set(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=df)
plt.savefig('Length_of_MembershipVsYearly_amount_spent.png')
X = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

y= df['Yearly Amount Spent']

#implement Linear regression model on our dataset
from sklearn.linear_model import LinearRegression


regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[20, 10, 30,2.06]]))
