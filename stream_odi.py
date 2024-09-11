import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
st.title("Prediction of ODI Match")
df=pd.read_csv("ODI_Match_info.csv")
new=df.dropna()
new['Team_1']=''
new['Team_2']=''
new['Decision']=''
new['Win']=''
label_encoder_team = preprocessing.LabelEncoder()
label_encoder_decision = preprocessing.LabelEncoder()
label_encoder_win = preprocessing.LabelEncoder()

new['Team_1'] = label_encoder_team.fit_transform(new['team1'])
new['Team_2'] = label_encoder_team.fit_transform(new['team2'])
new['Decision'] = label_encoder_decision.fit_transform(new['toss_decision'])
new['Win'] = label_encoder_win.fit_transform(new['winner'])

X=new.drop(['id','season','city','date'	,'team1','team2','toss_winner','toss_decision','result','dl_applied','win_by_wickets','player_of_match','venue'	,'umpire1','umpire2','umpire3','winner','win_by_runs','Win'],axis='columns')
y=new.drop(['id','season','city','date'	,'team1','team2','toss_winner','toss_decision','result','dl_applied','win_by_wickets','player_of_match','venue'	,'umpire1','umpire2','umpire3','Team_1','Team_2','Decision','winner','win_by_runs'],axis='columns')
X_train,X_test,y_train,y_test = train_test_split( X, y, test_size=0.2)
display=new.head(5)
st.write(display)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
t1 = st.text_input("Team 1", "Enter the name of Team 1")
t2 = st.text_input("Team 2", "Enter the name of Team 2")
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

team1_encoded =label_encoder_team.fit_transform([t1]) 
team2_encoded = label_encoder_team.fit_transform([t2]) 
toss_decision_encoded = label_encoder_decision.fit_transform([toss_decision]) 

input_data = np.array([team1_encoded[0], team2_encoded[0], toss_decision_encoded[0]]).reshape(1, -1)

if st.button("Predict"):
 prediction = model.predict(input_data)
 prediction_label = label_encoder_win.inverse_transform(prediction)
 st.write(f"Predicted Outcome: {prediction_label[0]}")

