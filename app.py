import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
from sklearn.preprocessing import OneHotEncoder 
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Sequential
import datetime


data=pd.read_csv("Churn_Modelling.csv")

##dropping down the columns that are not relavent

data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)

##Encoding categorical data

label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
onehot_encoder_geography=OneHotEncoder()
geography=onehot_encoder_geography.fit_transform(data[["Geography"]])
geography=geography.toarray()

##conert geography into data frame 

geography2=pd.DataFrame(geography,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

## dropping data not relevant and adding the new data that we use

data=data.drop('Geography',axis=1)
data=pd.concat([data,geography2],axis=1)
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)
with open("onehot_encoder_geography.pkl",'wb') as file:
    pickle.dump(onehot_encoder_geography,file)   

## spliting data into test and traing   
#                 
X=data.drop('Exited',axis=1)
Y=data['Exited']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

## scalering input

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#saving StandardScaler
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

##Building our ann model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ##HL1 connected with input layer
    Dense(32,activation='relu'),#HL2
    Dense(1,activation='sigmoid')#output layer
]

)

#complie the mode
opt= tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

#set up the tensorbord
log_dir='log/fit' + datetime.datetime.now().strftime("%Y%m%D-%H%m%S")
tensor_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
early_stoping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

#trainning the model
history=model.fit(
    X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,
    callbacks=[tensor_callback,early_stoping]
)
model.save("model.keras")

