import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import gdown

# --- Streamlit Input Widgets ---
player_name = st.text_input("Enter player name:")
pitch_type = st.text_input("Enter pitch type:")

# --- Load Data from Google Drive ---
@st.cache_data
def load_data_from_drive():
    # File ID from your Google Drive shareable link
    file_id = '1JxWQ8tTgVsxf9Y3-lsJ3qXFlH7a8ywmb'  # Replace with your actual file ID from the shareable link
    url = f'https://drive.google.com/uc?id={file_id}'

    # Download the CSV file
    output_path = '/tmp/sc.csv'
    gdown.download(url, output_path, quiet=False)

    # Load CSV into pandas DataFrame
    sc = pd.read_csv(output_path)
    return sc

sc = load_data_from_drive()

# --- Process and Show Data ---
if player_name and pitch_type:
    # Find Player's Pitch Data ---
    player_pitches = sc[(sc['player_name'] == player_name) & (sc['pitch_type'] == pitch_type)]

    if player_pitches.empty:
        st.write(f"No data found for player '{player_name}' and pitch type '{pitch_type}'.")
    else:
        avg_metrics = player_pitches[['arm_angle', 'release_speed', 'HB', 'iVB', 'release_spin_rate']].mean()
        st.write(f"\nAverage Pitch Metrics for {player_name} ({pitch_type}):")
        st.write(avg_metrics)

        # --- Find Similar Pitches ---
        features = ['arm_angle', 'release_speed', 'HB', 'iVB', 'release_spin_rate']
        player_pitch_vector = np.array(avg_metrics)

        # Handle potential missing values (NaN) by imputing with the mean of each feature.
        sc_filtered = sc[features].fillna(sc[features].mean())

        distances = cdist([player_pitch_vector], sc_filtered, metric='euclidean')
        sc['distance'] = distances[0]

        # Exclude the input player's pitches
        sc_filtered = sc[sc['player_name'] != player_name]

        # Top 10 most similar pitches (excluding the input pitch)
        top_10_similar = sc_filtered.sort_values('distance').head(10)

        st.write(f"\nTop 10 Most Similar Pitches:")
        st.write(top_10_similar[['player_name', 'pitch_type', 'arm_angle', 'release_speed', 'HB', 'iVB', 'release_spin_rate', 'distance']])