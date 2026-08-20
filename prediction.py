import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd

#load all models
model=load_model('model.keras')
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_geography=pickle.load(file)
with open('onehot_encoder_geography.pkl','rb')as file:
    onehot_encoder_geography=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

#EXample input data

input_data={
    'CreditScore':600,
    'Geography':'France',
    'Gender':'Male',
    'Age':40,
    'Tenure':3,
    'Balance':60000,
    'NumOfProducts':2,
    'HasCrCard':1,
    'IsActiveMember':1,
    'EstimatedSalary':50000

}
input_data_df=pd.DataFrame([input_data])
gro_encoded=onehot_encoder_geography.transform(input_data_df[['Geography']]).toarray()
gro_encoded_df=pd.DataFrame(gro_encoded,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_data_df['Gender']=label_encoder_geography.transform(input_data_df['Gender'])
input_data_df=input_data_df.drop('Geography',axis=1)
input_data_df=pd.concat([gro_encoded_df,input_data_df],axis=1)

#arranging column in sam eorder as in training

input_data_df=input_data_df[scaler.feature_names_in_]
scaler_input=scaler.transform(input_data_df)

#prediction#
predict=model.predict(scaler_input)
prob_predict=predict[0][0]
print(prob_predict)
if prob_predict >0.5:
    print("Customer is likely to churn")
else:
    print("customer is not likely to churn")