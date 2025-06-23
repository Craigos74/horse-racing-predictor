def scrape_race_data(url):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, 'html.parser')

    runner_rows = soup.select(".RC-runnerRow")
    horses = []
    field_size = len(runner_rows)

    for row in runner_rows:
        try:
            # Horse name
            name_tag = row.select_one(".RC-runnerName")
            horse = name_tag.text.strip() if name_tag else None

            # Official Rating (OR)
            or_tag = row.select_one(".RC-runnerRating")
            or_val = int(or_tag.text.strip()) if or_tag and or_tag.text.strip().isdigit() else 75

            # RPR (reuse OR if missing)
            rpr_tag = row.select_one(".RC-runnerRatingRC")
            rpr_last = int(rpr_tag.text.strip()) if rpr_tag and rpr_tag.text.strip().isdigit() else or_val
            rpr_prev = or_val

            # Weight
            wt_tag = row.select_one(".RC-horseWgt")
            weight_lbs = int(wt_tag.text.strip().split()[0]) if wt_tag else 126

            # Draw
            draw_tag = row.select_one(".RC-runnerDraw")
            draw = int(draw_tag.text.strip()) if draw_tag and draw_tag.text.strip().isdigit() else 5

            # Headgear
            hg_tag = row.select_one(".RC-runnerHeadgear")
            headgear = hg_tag.text.strip() if hg_tag else "None"

            # Add horse record
            if horse:
                horses.append({
                    'Horse': horse,
                    'OR': or_val,
                    'RPR_last': rpr_last,
                    'RPR_prev': rpr_prev,
                    'Weight_lbs': weight_lbs,
                    'Days_last': 21,  # placeholder
                    'Course_starts': 2,
                    'Course_wins': 1,
                    'Distance_wins': 1,
                    'Draw': draw,
                    'Going_pref': 'Good',
                    'Headgear': headgear,
                    'Prize': 20000,
                    'FieldSize': field_size
                })
        except Exception as e:
            print(f"Error parsing runner: {e}")
            continue

    return pd.DataFrame(horses)
