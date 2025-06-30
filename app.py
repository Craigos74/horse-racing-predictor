# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# Preprocessing Function
# -------------------------
def preprocess_race_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['or_diff'] = df['OR'] - df['OR'].mean()
    df['rpr_trend'] = df['RPR_last'] - df['RPR_prev']
    df['weight_adj'] = (df['Weight_lbs'] - 126) / 8
    df['course_sr'] = df['Course_wins'] / df['Course_starts'].replace(0, np.nan)
    df['course_sr'] = df['course_sr'].fillna(0)
    df['days_log'] = np.log1p(df['Days_last'])
    df['draw_scaled'] = (df['Draw'] - df['Draw'].mean()) / df['Draw'].std()
    df['distance_success'] = df['Distance_wins'] / (df['Distance_wins'] + 1)
    df['going_is_good'] = (df['Going_pref'] == 'Good').astype(int)
    df['prize_log'] = np.log1p(df['Prize'])
    df['field_scaled'] = df['FieldSize'] / df['FieldSize'].max()
    return df

# -------------------------
# Sample Training Data
# -------------------------
sample_data = {
    'race_id': [1, 1, 1, 2, 2, 2],
    'horse': ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E', 'Horse F'],
    'OR': [85, 82, 78, 70, 68, 65],
    'RPR_last': [88, 84, 77, 72, 70, 66],
    'RPR_prev': [86, 83, 76, 71, 69, 65],
    'Weight_lbs': [130, 128, 126, 122, 121, 120],
    'Days_last': [14, 21, 35, 7, 60, 12],
    'Course_starts': [2, 3, 1, 0, 1, 2],
    'Course_wins': [1, 0, 0, 0, 1, 0],
    'Distance_wins': [1, 1, 0, 0, 0, 0],
    'Draw': [1, 2, 3, 4, 5, 6],
    'Going_pref': ['Good', 'Firm', 'Good', 'Soft', 'Good', 'Firm'],
    'Headgear': ['None', 'Blinkers', 'Hood', 'None', 'Visor', 'None'],
    'Prize': [10000, 10000, 10000, 5000, 5000, 5000],
    'FieldSize': [8, 8, 8, 6, 6, 6],
    'Trainer': ['Trainer A', 'Trainer B', 'Trainer A', 'Trainer C', 'Trainer D', 'Trainer E'],
    'Jockey': ['Jockey A', 'Jockey B', 'Jockey A', 'Jockey C', 'Jockey D', 'Jockey E'],
    'Winner': [1, 0, 0, 0, 1, 0]
}

df_sample = pd.DataFrame(sample_data)
df_sample = preprocess_race_data(df_sample)

# -------------------------
# Model Training
# -------------------------
feature_cols = [
    'or_diff', 'rpr_trend', 'weight_adj', 'course_sr', 'days_log',
    'draw_scaled', 'distance_success', 'going_is_good', 'prize_log', 'field_scaled'
]

X_train = df_sample[feature_cols]
y_train = df_sample['Winner']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏇 Horse Racing Win Probability Predictor")
uploaded_file = st.file_uploader("📤 Upload a race card CSV", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    df_input = preprocess_race_data(df_input)

    # Handle missing or non-numeric data
    X_new = df_input[feature_cols].copy()
    X_new = X_new.fillna(0)
    X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict
    df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
    df_input['place_probability'] = df_input['win_probability'] * 1.6
    df_input['win_prob_%'] = (df_input['win_probability'] * 100).round(1).astype(str) + '%'
    df_input['place_prob_%'] = (df_input['place_probability'].clip(upper=1) * 100).round(1).astype(str) + '%'

    # Display Results
    st.subheader("📊 Predicted Results")
    st.dataframe(
        df_input[['Horse', 'win_probability', 'win_prob_%', 'place_probability', 'place_prob_%']]
        .sort_values(by='win_probability', ascending=False)
        .reset_index(drop=True)
    )
else:
    st.info("👆 Upload a CSV race card file to get started.")

