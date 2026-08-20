import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

model=load_model('model.keras')

with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geography.pkl','rb')as file:
    onehot_encoder_geography=pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

st.title("Customer Churn Prediction Project")

#user data
gender=st.selectbox('Gender',label_encoder_gender.classes_)
geography=st.selectbox('Geography',onehot_encoder_geography.categories_[0])
age=st.slider('Age',18,92,55)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimate_salary=st.number_input('Estimate Salary')
tenure=st.slider('Tenure',0,30)
number_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Had a credit card',[0,1])
is_active_member=st.selectbox('Is active member',[0,1])

#input data
input_data_df=pd.DataFrame(
    {
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        "NumOfProducts":[number_of_products],
        "HasCrCard":[has_cr_card],
        "IsActiveMember":[is_active_member],
        "EstimatedSalary":[estimate_salary]
    }
)
#should be copy paste form the traing data
geography=onehot_encoder_geography.transform([[geography]])
geography=geography.toarray()
geography2=pd.DataFrame(geography,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
data=pd.concat([input_data_df,geography2],axis=1)
scaler_input=scaler.transform(data)
predict=model.predict(scaler_input)
pro=predict[0][0]
st.write(f"Churn probability is: {pro}")
if pro>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer will not churn")