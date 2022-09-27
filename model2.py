

# Income orediction model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
iris_data=pd.read_excel("/home/Aiswaryatnair/iris (3).xls")
iris_data['Classification'].unique()
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
iris_data['Classification']= label_encoder.fit_transform(iris_data['Classification'])

iris_data['Classification'].unique()
x=iris_data.drop(['Classification'],axis=1)
y=iris_data['Classification']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
# we are taking the whole data to train model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#Fitting the model
m=regressor.fit(x_train,y_train)
#Saving the model to disk
pickle.dump(regressor,open('model2.pkl','wb') )
ypred = regressor.predict(x_test)
print("aaaa")
print(ypred)
print("aaaa")
print(x_test)
print(y_test)


