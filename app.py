import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Sample historical data to train the model
data = {
    'race_id': [1, 1, 1, 2, 2, 2],
    'horse': ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E', 'Horse F'],
    'OR': [85, 82, 78, 70, 68, 65],
    'RPR_last': [88, 84, 77, 72, 70, 66],
    'RPR_prev': [86, 83, 76, 71, 69, 65],
    'Weight_lbs': [130, 128, 126, 122, 121, 120],
    'Days_last': [14, 21, 35, 7, 60, 12],
    'Course_starts': [2, 3, 1, 0, 1, 2],
    'Course_wins': [1, 0, 0, 0, 1, 0],
    'Winner': [1, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
df['or_diff'] = df['OR'] - df.groupby('race_id')['OR'].transform('mean')
df['rpr_trend'] = df['RPR_last'] - df['RPR_prev']
df['weight_adj'] = (df['Weight_lbs'] - 126) / 8
df['course_sr'] = df['Course_wins'] / df['Course_starts'].replace(0, np.nan)
df['course_sr'] = df['course_sr'].fillna(0)
df['days_log'] = np.log1p(df['Days_last'])

feature_cols = ['or_diff', 'rpr_trend', 'weight_adj', 'course_sr', 'days_log']
X = df[feature_cols]
y = df['Winner']

scaler = StandardScaler()
model = LogisticRegression()
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', model)
])
pipeline.fit(X, y)

# Streamlit App
st.title("Horse Racing Win Predictor")
st.write("Upload a race card CSV or enter horse data manually:")

# CSV Upload Section
uploaded_file = st.file_uploader("Upload CSV file with horse race data", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    df_input['or_diff'] = df_input['OR'] - df_input['OR'].mean()
    df_input['rpr_trend'] = df_input['RPR_last'] - df_input['RPR_prev']
    df_input['weight_adj'] = (df_input['Weight_lbs'] - 126) / 8
    df_input['course_sr'] = df_input['Course_wins'] / df_input['Course_starts'].replace(0, np.nan)
    df_input['course_sr'] = df_input['course_sr'].fillna(0)
    df_input['days_log'] = np.log1p(df_input['Days_last'])

    X_new = df_input[feature_cols]
    df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
    df_input['place_probability'] = df_input['win_probability'] * 1.6

    st.subheader("Predicted Results from CSV")
    st.dataframe(df_input[['Horse', 'win_probability', 'place_probability']].sort_values(by='win_probability', ascending=False))

else:
    num_horses = st.number_input("Number of horses", min_value=2, max_value=12, value=3)
    input_data = []
    for i in range(num_horses):
        st.subheader(f"Horse {i+1}")
        horse = st.text_input(f"Name of Horse {i+1}", key=f"name_{i}")
        or_rating = st.number_input(f"Official Rating (OR) - {horse}", key=f"or_{i}")
        rpr_last = st.number_input(f"RPR Last Run - {horse}", key=f"rprl_{i}")
        rpr_prev = st.number_input(f"RPR Previous Run - {horse}", key=f"rprp_{i}")
        weight = st.number_input(f"Weight (lbs) - {horse}", key=f"wt_{i}")
        days_last = st.number_input(f"Days Since Last Run - {horse}", key=f"days_{i}")
        starts = st.number_input(f"Course Starts - {horse}", key=f"starts_{i}")
        wins = st.number_input(f"Course Wins - {horse}", key=f"wins_{i}")

        input_data.append({
            'Horse': horse,
            'OR': or_rating,
            'RPR_last': rpr_last,
            'RPR_prev': rpr_prev,
            'Weight_lbs': weight,
            'Days_last': days_last,
            'Course_starts': starts,
            'Course_wins': wins
        })

    if st.button("Predict Win Chances"):
        df_input = pd.DataFrame(input_data)
        df_input['or_diff'] = df_input['OR'] - df_input['OR'].mean()
        df_input['rpr_trend'] = df_input['RPR_last'] - df_input['RPR_prev']
        df_input['weight_adj'] = (df_input['Weight_lbs'] - 126) / 8
        df_input['course_sr'] = df_input['Course_wins'] / df_input['Course_starts'].replace(0, np.nan)
        df_input['course_sr'] = df_input['course_sr'].fillna(0)
        df_input['days_log'] = np.log1p(df_input['Days_last'])

        X_new = df_input[feature_cols]
        df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
        df_input['place_probability'] = df_input['win_probability'] * 1.6

        st.subheader("Predicted Results")
        st.dataframe(df_input[['Horse', 'win_probability', 'place_probability']].sort_values(by='win_probability', ascending=False))
