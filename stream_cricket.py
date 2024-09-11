import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree

st.title("Prediction of ODI Match")

# Load dataset and preprocess
df = pd.read_csv("ODI_Match_info.csv")
new = df.dropna()

# Initialize new columns
new['Team_1'] = ''
new['Team_2'] = ''
new['Decision'] = ''
new['Win'] = ''

# Initialize label encoders
label_encoder_team = preprocessing.LabelEncoder()
label_encoder_decision = preprocessing.LabelEncoder()
label_encoder_win = preprocessing.LabelEncoder()

# Fit label encoders
label_encoder_team.fit(pd.concat([new['team1'], new['team2'], new['winner']]))
label_encoder_decision.fit(new['toss_decision'])
label_encoder_win.fit(new['winner'])

# Transform columns using fitted label encoders
new['Team_1'] = label_encoder_team.transform(new['team1'])
new['Team_2'] = label_encoder_team.transform(new['team2'])
new['Decision'] = label_encoder_decision.transform(new['toss_decision'])
new['Win'] = label_encoder_team.transform(new['winner'])

# Define features and target variable
X = new.drop(['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner', 'toss_decision', 'result', 'dl_applied', 'win_by_wickets', 'player_of_match', 'venue', 'umpire1', 'umpire2', 'umpire3', 'winner', 'win_by_runs', 'Win'], axis='columns')
y = new['Win']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# Display sample data
st.write("Sample Data:")
display = new.head(5)
st.write(display)

# Inputs for prediction
t1 = st.text_input("Team 1", "Enter the name of Team 1")
t2 = st.text_input("Team 2", "Enter the name of Team 2")
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# Encode inputs with error handling
def encode_label(label, encoder):
    try:
        return encoder.transform([label])[0]
    except ValueError:
        st.write(f"Error: The label '{label}' is not in the training data.")
        return None

team1_encoded = encode_label(t1, label_encoder_team)
team2_encoded = encode_label(t2, label_encoder_team)
toss_decision_encoded = encode_label(toss_decision, label_encoder_decision)

# Prepare input data and make prediction
if team1_encoded is not None and team2_encoded is not None and toss_decision_encoded is not None:
    input_data = np.array([team1_encoded, team2_encoded, toss_decision_encoded]).reshape(1, -1)
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_label = label_encoder_team.inverse_transform(prediction)
        st.write(f"Predicted Outcome: {prediction_label[0]}")

