###########
"""
Author: Lui Go Nam
Version:0.2
Function: Turn parsed METAR data into airport weather score (Europe airport)

"""

import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
###########

# WIND SPEED(KT) TO WEATHER SCORE
# Input: wind speed(kt)
# Output: score


def get_wind_intensity_code(spd, spd_gust):
    if spd is None:
        return 0

    if spd <= 15:
        answer = 0
    elif 15 < spd <= 20:
        answer = 1
    elif 20 < spd <= 30:
        answer = 2
    elif spd > 30:
        answer = 3
    else:
        answer = 0

    if spd_gust is not None:
        answer += 1

    return answer


# DANGEROUS PHENOMENON TO WEATHER SCORE
# Input: Unparsed METAR code (str)
# Output: score
def get_convect_weather_code(
    present_weather,
    cloud_cover1,
    cloud_type1,
    cloud_cover2,
    cloud_type2,
    cloud_cover3,
    cloud_type3,
):
    dangerous_phenomena = 0
    cb = 0
    tcu = 0
    ts_hint = 0

    if present_weather is None:
        present_weather = ""

    # Dangerous phenomena
    if "FC" in present_weather:
        dangerous_phenomena = 24
    elif "DS" in present_weather:
        dangerous_phenomena = 24
    elif "SS" in present_weather:
        dangerous_phenomena = 24
    elif "VA" in present_weather:
        dangerous_phenomena = 24
    elif "SA" in present_weather:
        dangerous_phenomena = 24
    elif "SQ" in present_weather:
        dangerous_phenomena = 24
    elif "GS" in present_weather:
        dangerous_phenomena = 18
    elif "GR" in present_weather:
        dangerous_phenomena = 24
    elif "PL" in present_weather:
        dangerous_phenomena = 24
    elif "TS" in present_weather and "+" in present_weather:
        dangerous_phenomena = 30
    elif "TS" in present_weather:
        dangerous_phenomena = 24

    # Cumulonimbus
    for cover, type in [
        (cloud_cover1, cloud_type1),
        (cloud_cover2, cloud_type2),
        (cloud_cover3, cloud_type3),
    ]:
        if cover == "OVC" and type == "CB":
            cb = 12
        elif cover == "BKN" and type == "CB":
            cb = 10
        elif cover == "SCT" and type == "CB":
            cb = 6
        elif cover == "FEW" and type == "CB":
            cb = 4

    # Towering Cumulus
    for cover, type in [
        (cloud_cover1, cloud_type1),
        (cloud_cover2, cloud_type2),
        (cloud_cover3, cloud_type3),
    ]:
        if cover == "OVC" and type == "TCU":
            tcu = 10
        elif cover == "BKN" and type == "TCU":
            tcu = 8
        elif cover == "SCT" and type == "TCU":
            tcu = 5
        elif cover == "FEW" and type == "TCU":
            tcu = 3

    # Thunderstorm hint
    if cb == 12 and "-SH" in present_weather:
        ts_hint = 18
    elif (cb == 10 or tcu == 10) and "-SH" in present_weather:
        ts_hint = 12
    elif (cb == 6 or tcu == 8) and "-SH" in present_weather:
        ts_hint = 10
    elif (cb == 4 or tcu == 5) and "-SH" in present_weather:
        ts_hint = 8
    elif tcu == 3 and "-SH" in present_weather:
        ts_hint = 4
    elif cb == 12 and "SH" in present_weather:
        ts_hint = 24
    elif (cb == 10 or tcu == 10) and "SH" in present_weather:
        ts_hint = 20
    elif (cb == 6 or tcu == 8) and "SH" in present_weather:
        ts_hint = 15
    elif (cb == 4 or tcu == 5) and "SH" in present_weather:
        ts_hint = 12
    elif tcu == 3 and "SH" in present_weather:
        ts_hint = 6

    return max(dangerous_phenomena, cb, tcu, ts_hint)


def get_precipitations_code(present_weather):
    if present_weather is None:
        return 0

    if "FZ" in present_weather:
        return 3
    elif "-SN" in present_weather:
        return 2
    elif "SN" in present_weather:
        return 3
    elif "SG" in present_weather:
        return 2
    elif "+RA" in present_weather:
        return 2
    elif "+SHRA" in present_weather:
        return 2
    elif "-RA" in present_weather:
        return 0
    elif "IC" in present_weather:
        return 1
    elif "RA" in present_weather:
        return 1
    elif "UP" in present_weather:
        return 1
    elif "DZ" in present_weather:
        return 1
    else:
        return 0


def get_visi_ceiling_code(
    vis1,
    vis2,
    rvr1,
    rvr2,
    cld_cover1,
    cld_cover2,
    cld_cover3,
    cld_base1,
    cld_base2,
    cld_base3,
):
    cte_vis = min(vis1 or float("inf"), vis2 or float("inf"))
    cte_rvr = min(rvr1 or float("inf"), rvr2 or float("inf"))
    cte_cld_base = min(
        cld_base1 or float("inf"), cld_base2 or float("inf"), cld_base3 or float("inf")
    )
    cte_cld_cover = (
        1
        if "BKN" in (cld_cover1, cld_cover2, cld_cover3)
        or "OVC" in (cld_cover1, cld_cover2, cld_cover3)
        else 0
    )

    if (cte_rvr <= 325) or (cte_cld_cover == 1 and cte_cld_base <= 50):
        return 5
    elif (350 <= cte_rvr <= 500) or (cte_cld_cover == 1 and 100 <= cte_cld_base <= 150):
        return 4
    elif (550 <= cte_rvr <= 750) or (cte_cld_cover == 1 and 200 <= cte_cld_base <= 250):
        return 2
    else:
        return 0


def get_fz_conditions_code(t, td, present_weather):
    if t is None or td is None:
        return 0

    cte_visible_moisture = 0
    if present_weather:
        if "FZRA" in present_weather:
            cte_visible_moisture = 5
        elif "+RA" in present_weather:
            cte_visible_moisture = 4
        elif "SG" in present_weather:
            cte_visible_moisture = 4
        elif "RASN" in present_weather:
            cte_visible_moisture = 4
        elif "-SN" in present_weather:
            cte_visible_moisture = 4
        elif "SN" in present_weather:
            cte_visible_moisture = 5
        elif "BR" in present_weather:
            cte_visible_moisture = 4
        elif "RA" in present_weather:
            cte_visible_moisture = 3
        elif "PL" in present_weather:
            cte_visible_moisture = 3
        elif "IC" in present_weather:
            cte_visible_moisture = 3
        elif "GR" in present_weather:
            cte_visible_moisture = 3
        elif "GS" in present_weather:
            cte_visible_moisture = 3
        elif "UP" in present_weather:
            cte_visible_moisture = 3
        elif "FG" in present_weather:
            cte_visible_moisture = 3
        elif "DZ" in present_weather:
            cte_visible_moisture = 3
        else:
            cte_visible_moisture = 0
    else:
        cte_visible_moisture = None

    cte_temperature_dew = t - td

    if t <= 3 and cte_visible_moisture == 5:
        return 4
    elif t < -15 and cte_visible_moisture is not None:
        return 4
    elif t <= 3 and cte_visible_moisture == 4:
        return 3
    elif t <= 3 and (cte_visible_moisture == 3 or cte_temperature_dew < 3):
        return 1
    elif t <= 3 and cte_visible_moisture is None:
        return 0
    elif t > 3 and cte_visible_moisture is not None and cte_visible_moisture > 0:
        return 0
    elif t > 3 and (cte_visible_moisture is None or cte_temperature_dew >= 3):
        return 0
    else:
        return 0


def metar_to_scores(metar_text):
    def extract_value(pattern, text):
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def safe_int(value, default=None):
        try:
            return int(value) if value is not None else default
        except ValueError:
            return default

    lines = metar_text.strip().split('\n')
    results = []

    for line in lines:
        try:
            # Split the line into timestamp and METAR report
            timestamp, metar = line.split(' METAR ', 1)
            
            # Parse the timestamp
            date_time = datetime.strptime(timestamp, '%Y%m%d%H%M')

            wind_speed = safe_int(extract_value(r'(\d{2})KT', metar))
            wind_gust = safe_int(extract_value(r'(\d{2})G(\d{2})KT', metar))
            visibility = safe_int(extract_value(r'\s(\d{4})\s', metar))
            temp = safe_int(extract_value(r'\s(\d{2})/\d{2}\s', metar))
            dew_point = safe_int(extract_value(r'\s\d{2}/(\d{2})\s', metar))
            
            cloud_match = re.search(r'(FEW|SCT|BKN|OVC)(\d{3})', metar)
            cloud_cover = cloud_match.group(1) if cloud_match else None
            cloud_base = safe_int(cloud_match.group(2), 999) * 100 if cloud_match else None

            wind_score = get_wind_intensity_code(wind_speed, wind_gust)
            
            dangerous_phenom_score = get_convect_weather_code(metar, 
                                                              cloud_cover, 'CB' if 'CB' in metar else None,
                                                              None, None, None, None)
            
            precip_score = get_precipitations_code(metar)
            
            vis_ceiling_score = get_visi_ceiling_code(visibility, None, None, None, 
                                                      cloud_cover, None, None,
                                                      cloud_base, None, None)
            
            freezing_score = get_fz_conditions_code(temp, dew_point, metar)

            total_score = wind_score + dangerous_phenom_score + precip_score + vis_ceiling_score + freezing_score

            results.append({
                'datetime': date_time,
                'wind_score': wind_score,
                'dangerous_phenom_score': dangerous_phenom_score,
                'precip_score': precip_score,
                'vis_ceiling_score': vis_ceiling_score,
                'freezing_score': freezing_score,
                'total_score': total_score
            })
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Error message: {str(e)}")

    return results

def metar_to_csv(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as file:
        metar_data = file.read()

    # Process the METAR data
    scores = metar_to_scores(metar_data)

    # Write the results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['datetime', 'wind_score', 'dangerous_phenom_score', 
                      'precip_score', 'vis_ceiling_score', 'freezing_score', 'total_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for score in scores:
            # Convert datetime to string for CSV writing
            score['datetime'] = score['datetime'].strftime('%Y-%m-%d %H:%M')
            writer.writerow(score)

    print(f"CSV file '{output_file}' has been created successfully.")

def visualize_metar_scores(csv_file, output_image):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert datetime strings to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort the dataframe by datetime
    df = df.sort_values('datetime')
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot each score
    plt.plot(df['datetime'], df['wind_score'], c='#c03d3e', linewidth=0.8)
    plt.plot(df['datetime'], df['vis_ceiling_score'], c='#3a923a', linewidth=0.8)
    plt.plot(df['datetime'], df['precip_score'], c='#3274a1', linewidth=0.8)
    plt.plot(df['datetime'], df['freezing_score'], c='m', linewidth=0.8)
    plt.scatter(df['datetime'], df['dangerous_phenom_score'], c='k', s=10, marker='+')
    
    # Create legend patches
    legend1 = mpatches.Patch(color='#c03d3e', label='Wind')
    legend2 = mpatches.Patch(color='#3a923a', label='Visibility')
    legend3 = mpatches.Patch(color='#3274a1', label='Precipitation')
    legend4 = mpatches.Patch(color='m', label='Freeze condition')
    legend5 = mpatches.Patch(color='k', label='Dangerous phenomenon')
    
    # Add legend
    plt.legend(handles=[legend1, legend3, legend2, legend4, legend5], loc='best', fontsize=12)
    
    # Set y-axis limits
    plt.ylim(0, 22)
    
    # Set x-axis locator
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    
    # Set labels and title
    plt.ylabel('Score', fontsize=14)
    plt.title('METAR Scores Over Time', fontsize=16)
    
    # Rotate x-axis labels
    plt.xticks(rotation=20, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid
    plt.grid(True)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_image}")
    
    # Close the plot to free up memory
    plt.close()