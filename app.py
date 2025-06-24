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
            name_tag = row.select_one(".RC-runnerName a, .RC-runnerName")
            horse = name_tag.get_text(strip=True) if name_tag else "Unknown"

            # Official Rating
            or_val = 75  # Default fallback
            rating_tags = row.select(".RC-horseInfo span")
            for span in rating_tags:
                if 'OR' in span.text:
                    try:
                        or_val = int(span.text.split()[-1])
                    except:
                        pass
                    break

            # RPR - use OR as fallback
            rpr_last = or_val + 2
            rpr_prev = or_val

            # Weight
            weight_tag = row.select_one(".RC-runnerWgt, .RC-horseWgt")
            try:
                weight_lbs = int(weight_tag.text.strip().split()[0]) if weight_tag else 126
            except:
                weight_lbs = 126

            # Draw
            draw_tag = row.select_one(".RC-runnerDraw")
            try:
                draw = int(draw_tag.text.strip().replace('(', '').replace(')', '')) if draw_tag else 5
            except:
                draw = 5

            # Headgear
            headgear_tag = row.select_one(".RC-runnerHeadgear, .RC-headgear")
            headgear = headgear_tag.get_text(strip=True) if headgear_tag else "None"

            # Append to list
            horses.append({
                'Horse': horse,
                'OR': or_val,
                'RPR_last': rpr_last,
                'RPR_prev': rpr_prev,
                'Weight_lbs': weight_lbs,
                'Days_last': 21,
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
            print(f"[Scraper Error] Failed to parse runner: {e}")
            continue

    return pd.DataFrame(horses)
