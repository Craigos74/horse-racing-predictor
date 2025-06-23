import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import requests
from bs4 import BeautifulSoup

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

def scrape_race_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    runners = soup.select(".RC-runnerRow")
    horses = []
    field_size = len(runners)

    for row in runners:
        try:
            horse = row.select_one(".RC-runnerName").get_text(strip=True)
            or_tags = row.select(".RC-horseInfo span")
            or_val = 75
            for span in or_tags:
                if 'OR' in span.get_text():
                    or_val = int(span.get_text().split()[-1])
                    break
            weight = row.select_one(".RC-runnerWgt")
            weight_lbs = int(weight.get_text(strip=True).split()[0]) if weight else 126
            draw = row.select_one(".RC-runnerDraw")
            draw_num = int(draw.get_text(strip=True).replace('(', '').replace(')', '')) if draw else 5
            headgear = row.select_one(".RC-headgear")
            headgear_val = headgear.get_text(strip=True) if headgear else 'None'

            horses.append({
                'Horse': horse,
                'OR': or_val,
                'RPR_last': or_val + 2,
                'RPR_prev': or_val,
                'Weight_lbs': weight_lbs,
                'Days_last': 21,
                'Course_starts': 2,
                'Course_wins': 1,
                'Distance_wins': 1,
                'Draw': draw_num,
                'Going_pref': 'Good',
                'Headgear': headgear_val,
                'Prize': 20000,
                'FieldSize': field_size
            })
        except Exception as e:
            print(f"Error parsing runner: {e}")
            continue

    df_scraped = pd.DataFrame(horses)
    return df_scraped

# Sample historical data to train the model
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

df = pd.DataFrame(sample_data)
df = preprocess_race_data(df)

feature_cols = [
    'or_diff', 'rpr_trend', 'weight_adj', 'course_sr', 'days_log',
    'draw_scaled', 'distance_success', 'going_is_good', 'prize_log', 'field_scaled'
]

X = df[feature_cols]
y = df['Winner']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X, y)

# Streamlit App
st.title("Horse Racing Win Predictor")
st.write("Upload a race card CSV, enter data manually, or fetch race by URL:")

url_input = st.text_input("Paste racecard URL to fetch data (e.g. AtTheRaces or IrishRacing):")
if url_input and st.button("Fetch Race Data"):
    df_input = scrape_race_data(url_input)
    required_cols = {'OR', 'RPR_last', 'RPR_prev', 'Weight_lbs', 'Days_last', 'Course_starts', 'Course_wins', 'Distance_wins', 'Draw', 'Going_pref', 'Prize', 'FieldSize'}
    if not required_cols.issubset(df_input.columns):
        st.warning("Some fields are missing. Applying fallback default values.")
        for col in required_cols:
            if col not in df_input.columns:
                default_val = 75 if 'OR' in col or 'RPR' in col else 1
                if col in ['Weight_lbs']: default_val = 126
                if col in ['Draw']: default_val = 5
                if col in ['Going_pref']: default_val = 'Good'
                if col in ['Prize']: default_val = 20000
                if col in ['FieldSize']: default_val = len(df_input)
                df_input[col] = default_val
        df_input = preprocess_race_data(df_input)
    else:
        df_input = preprocess_race_data(df_input)
    X_new = pd.DataFrame()
    if not df_input.empty:
        missing_features = [col for col in feature_cols if col not in df_input.columns]
        if missing_features:
            st.error(f"Missing features in data: {missing_features}. Prediction skipped.")
        else:
            X_new = df_input[feature_cols]
            df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
            df_input['place_probability'] = df_input['win_probability'] * 1.6
            df_input['win_prob_%'] = (df_input['win_probability'] * 100).round(1).astype(str) + '%'
            df_input['place_prob_%'] = (df_input['place_probability'].clip(upper=1) * 100).round(1).astype(str) + '%'
            st.subheader("Predicted Results from URL")
            st.dataframe(df_input[['Horse', 'win_probability', 'win_prob_%', 'place_probability', 'place_prob_%']].sort_values(by='win_probability', ascending=False))
    df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
    df_input['place_probability'] = df_input['win_probability'] * 1.6
    df_input['win_prob_%'] = (df_input['win_probability'] * 100).round(1).astype(str) + '%'
    df_input['place_prob_%'] = (df_input['place_probability'].clip(upper=1) * 100).round(1).astype(str) + '%'
    st.subheader("Predicted Results from URL")
    st.dataframe(df_input[['Horse', 'win_probability', 'win_prob_%', 'place_probability', 'place_prob_%']].sort_values(by='win_probability', ascending=False))
else:
    st.error("Race data could not be processed due to missing or invalid features.")

# CSV Upload Section
uploaded_file = st.file_uploader("Upload CSV file with horse race data", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    df_input = preprocess_race_data(df_input)
    X_new = df_input[feature_cols]
    df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
    df_input['place_probability'] = df_input['win_probability'] * 1.6
    df_input['win_prob_%'] = (df_input['win_probability'] * 100).round(1).astype(str) + '%'
    df_input['place_prob_%'] = (df_input['place_probability'].clip(upper=1) * 100).round(1).astype(str) + '%'
    st.subheader("Predicted Results from CSV")
    st.dataframe(df_input[['Horse', 'win_probability', 'win_prob_%', 'place_probability', 'place_prob_%']].sort_values(by='win_probability', ascending=False))

# Manual Entry Section
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
        dist_wins = st.number_input(f"Distance Wins - {horse}", key=f"distwins_{i}")
        draw = st.number_input(f"Draw (Stall) - {horse}", key=f"draw_{i}")
        going_pref = st.selectbox(f"Going Preference - {horse}", ["Good", "Firm", "Soft", "Heavy"], key=f"going_{i}")
        headgear = st.text_input(f"Headgear - {horse}", key=f"gear_{i}")
        prize = st.number_input(f"Prize Money - {horse}", key=f"prize_{i}")
        field_size = st.number_input(f"Field Size - {horse}", key=f"field_{i}")

        input_data.append({
            'Horse': horse,
            'OR': or_rating,
            'RPR_last': rpr_last,
            'RPR_prev': rpr_prev,
            'Weight_lbs': weight,
            'Days_last': days_last,
            'Course_starts': starts,
            'Course_wins': wins,
            'Distance_wins': dist_wins,
            'Draw': draw,
            'Going_pref': going_pref,
            'Headgear': headgear,
            'Prize': prize,
            'FieldSize': field_size
        })

    if st.button("Predict Win Chances"):
        df_input = pd.DataFrame(input_data)
        df_input = preprocess_race_data(df_input)
        X_new = df_input[feature_cols]
        df_input['win_probability'] = pipeline.predict_proba(X_new)[:, 1]
        df_input['place_probability'] = df_input['win_probability'] * 1.6
        df_input['win_prob_%'] = (df_input['win_probability'] * 100).round(1).astype(str) + '%'
        df_input['place_prob_%'] = (df_input['place_probability'].clip(upper=1) * 100).round(1).astype(str) + '%'
        st.subheader("Predicted Results")
        st.dataframe(df_input[['Horse', 'win_probability', 'win_prob_%', 'place_probability', 'place_prob_%']].sort_values(by='win_probability', ascending=False))



