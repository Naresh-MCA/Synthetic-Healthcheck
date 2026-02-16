import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv('synthetic_health_data.csv')

cols=df.select_dtypes(include=['int','float64']).columns
q1=df[cols].quantile(0.25)
q3=df[cols].quantile(0.75)
IQR=q3-q1
lower=q1-1.5*IQR
upper=q3+1.5*IQR
df = df[~((df[cols] < (q1 - 1.5 * IQR)) |
          (df[cols] > (q3 + 1.5 * IQR))).any(axis=1)]

x=df.drop(['Health_Score'],axis=1)
y=df['Health_Score']

scaler=StandardScaler()
scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scaler=StandardScaler()
#x_fit=scaler.fit_transform(x_train)
#xtest_fit=scaler.fit_transform(x_test)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error, r2_score

#y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_pred,y_test))
print("R2 Score:", r2_score(y_pred,y_test))


import pickle
with open('synthetic_health.pkl','wb') as f:
    pickle.dump(model,f)

#pickle.dump(model, open("synthetic_health.pkl", "wb"))
#pickle.dump(scaler, open("scaler.pkl", "wb"))
