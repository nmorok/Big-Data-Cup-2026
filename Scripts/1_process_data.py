import pandas as pd
import numpy as np
import os
import glob
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Silence the Pandas future warning
pd.set_option('future.no_silent_downcasting', True)

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
#OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'Cluster_Time_Series_Activation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30
DT = 1.0 / FPS
DOWNSAMPLE_RATE = 15 # 2 Hz (Every 0.5 seconds)
N_CLUSTERS = 6
MAX_VELOCITY = 38.28
MAX_ACCELERATION = 16.40
MAX_DECELERATION = -50.0

ACTIVE_CLUSTERS = [0, 1, 3, 5]

def extract_period_number(p_val):
    """Safely extracts integer period, converting 'OT' to 4"""
    val_str = str(p_val).upper().strip()
    if val_str == 'OT' or val_str == 'POT':
        return 4
    try:
        match = re.search(r'\d+', val_str)
        return int(match.group()) if match else 1
    except:
        return 1

# ==========================================
# 1. TIME-SERIES FEATURE ENGINEERING
# ==========================================
def extract_time_series_features(tracking_file):
    df_raw = pd.read_csv(tracking_file, low_memory=False)

    filename = os.path.basename(tracking_file)
    game_date = filename.split('.')[0]

    df_raw['Frame_Number'] = df_raw['Image Id'].str.split('_').str[-1].astype(int)
    df_raw = df_raw.rename(columns={'Rink Location X (Feet)': 'X', 'Rink Location Y (Feet)': 'Y'})

    # --- THE OT FIX: Clean the Period column before doing math ---
    df_raw['Period'] = df_raw['Period'].apply(extract_period_number)

    def parse_clock(c):
        try:
            m, s = str(c).strip().split(':')
            return int(m) * 60 + int(s)
        except:
            return np.nan

    clock_col = 'Game Clock' if 'Game Clock' in df_raw.columns else 'Clock'
    if clock_col in df_raw.columns:
        df_raw['Seconds_Remaining'] = df_raw[clock_col].apply(parse_clock)
        df_raw['Period_Elapsed'] = 1200 - df_raw['Seconds_Remaining']
        df_raw['True_Game_Elapsed'] = (df_raw['Period'] - 1) * 1200 + df_raw['Period_Elapsed']
    else:
        df_raw['True_Game_Elapsed'] = np.nan

    match = re.search(r'(Team\.[A-Z]).*@.*(Team\.[A-Z])', filename, re.IGNORECASE)
    if match:
        away_team = match.group(1).replace('.', ' ')
        home_team = match.group(2).replace('.', ' ')
    else:
        away_team, home_team = "Away", "Home"

    puck_df = df_raw[df_raw['Player or Puck'] == 'Puck'][['Period', 'Frame_Number', 'X', 'Y']]
    puck_df = puck_df.rename(columns={'X': 'Puck_X', 'Y': 'Puck_Y'})

    df = df_raw[df_raw['Player or Puck'] == 'Player'].copy()
    df = df[df['Player Jersey Number'] != 'Go'].dropna(subset=['Player Jersey Number'])

    team_mapping = {'Away': away_team, 'Home': home_team}
    df['Actual_Team'] = df['Team'].map(team_mapping).fillna(df['Team'])
    df['Unique_Player_ID'] = df['Actual_Team'] + '_#' + df['Player Jersey Number'].astype(int).astype(str)
    df['Game_ID'] = game_date + "_" + away_team + "_at_" + home_team

    df = df.merge(puck_df, on=['Period', 'Frame_Number'], how='inner')
    df = df.sort_values(['Unique_Player_ID', 'Period', 'Frame_Number']).reset_index(drop=True)

    del df_raw, puck_df
    gc.collect()

    team_centroids = df.groupby(['Period', 'Frame_Number', 'Team'])[['X', 'Y']].transform('mean')
    df['Dist_to_Centroid'] = np.hypot(df['X'] - team_centroids['X'], df['Y'] - team_centroids['Y'])
    df['Dist_to_Net'] = np.minimum(np.hypot(df['X'] - 89, df['Y']), np.hypot(df['X'] + 89, df['Y']))
    df['In_Crease'] = (df['Dist_to_Net'] < 8.0).astype(int)
    df['In_Slot'] = (df['Dist_to_Net'] < 20.0).astype(int)
    df['Dist_to_Puck'] = np.hypot(df['X'] - df['Puck_X'], df['Y'] - df['Puck_Y'])
    df['Around_Puck'] = (df['Dist_to_Puck'] < 5.0).astype(int)

    group = df.groupby(['Unique_Player_ID', 'Period'])
    df['Frame_Diff'] = group['Frame_Number'].diff()
    dt_series = df['Frame_Diff'] * DT

    df['Velocity'] = np.hypot(group['X'].diff(), group['Y'].diff()) / dt_series
    df['Velocity'] = group['Velocity'].transform(lambda x: x.ffill().rolling(9, center=True, min_periods=1).mean())
    df.loc[(df['Velocity'] >= MAX_VELOCITY) | (df['Velocity'] <= 0), 'Velocity'] = np.nan

    df['Acceleration'] = (group['Velocity'].diff() / dt_series)
    df.loc[(df['Acceleration'] >= MAX_ACCELERATION) | (df['Acceleration'] <= MAX_DECELERATION), 'Acceleration'] = np.nan

    df['Abs_Acceleration'] = df['Acceleration'].abs()
    df['Abs_Acceleration'] = df.groupby(['Unique_Player_ID', 'Period'])['Abs_Acceleration'].transform(lambda x: x.ffill().rolling(9, center=True, min_periods=1).mean())
    df['Jerk'] = (df.groupby(['Unique_Player_ID', 'Period'])['Abs_Acceleration'].diff() / dt_series).abs()
    df['Jerk'] = df.groupby(['Unique_Player_ID', 'Period'])['Jerk'].transform(lambda x: x.ffill().rolling(9, center=True, min_periods=1).mean())

    df['Closing_Speed_on_Puck'] = -(group['Dist_to_Puck'].diff() / dt_series)

    # Replaced inplace=True to be fully compliant with Pandas 3.0 standards
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Velocity', 'Abs_Acceleration', 'Jerk', 'Closing_Speed_on_Puck']).copy()

    df['Time_Bucket'] = df['Frame_Number'] // DOWNSAMPLE_RATE
    ts_df = df.groupby(['Game_ID', 'Actual_Team', 'Unique_Player_ID', 'Period', 'Time_Bucket']).mean(numeric_only=True).reset_index()

    return ts_df

# ==========================================
# 2. STATE CLUSTERING & ACTIVATION SCORING
# ==========================================
def generate_activation_series(master_ts_df):
    print(f"\nClustering {len(master_ts_df)} micro-states across all players...")

    features = [
        'Velocity', 'Abs_Acceleration', 'Jerk',
        'Dist_to_Puck', 'Closing_Speed_on_Puck',
        'Dist_to_Centroid', 'Dist_to_Net', 'In_Crease',
        'In_Slot', 'Around_Puck'
    ]

    X = master_ts_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
    master_ts_df['State_Cluster'] = kmeans.fit_predict(X_scaled)

    # --- RESTORED: Output Cluster Profiles for the "Eye Test" ---
    cluster_means = master_ts_df.groupby('State_Cluster')[features].mean()
    print("\n=== CLUSTER PROFILES ===")
    print(cluster_means.round(2))

    plt.figure(figsize=(12, 6))
    scaled_means = pd.DataFrame(scaler.fit_transform(cluster_means), columns=features, index=cluster_means.index)
    sns.heatmap(scaled_means, annot=True, cmap='coolwarm', center=0)
    plt.title('Micro-State Tactical Heatmap')
    plt.ylabel('Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Micro_State_Profiles.png'))
    plt.close()
    print(f"\nSaved cluster heatmap to {OUTPUT_DIR}/Micro_State_Profiles.png")
    # -------------------------------------------------------------

    master_ts_df['Is_Active'] = master_ts_df['State_Cluster'].isin(ACTIVE_CLUSTERS).astype(int)

    return master_ts_df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    tracking_files = glob.glob(os.path.join(BASE_DATA_DIR, "**", "*Tracking_P*.csv"), recursive=True)

    # Also grab OT files explicitly if they exist
    ot_files = glob.glob(os.path.join(BASE_DATA_DIR, "**", "*Tracking_POT*.csv"), recursive=True)
    all_files = tracking_files + ot_files

    all_time_series = []

    for f in all_files:
        print(f"Processing Series: {os.path.basename(f)}")
        try:
            ts_data = extract_time_series_features(f)
            if not ts_data.empty:
                all_time_series.append(ts_data)
            del ts_data
            gc.collect()
        except Exception as e:
            print(f"  Error processing {os.path.basename(f)}: {e}")

    if all_time_series:
        master_df = pd.concat(all_time_series)
        final_ts_dataset = generate_activation_series(master_df)

        output_path = os.path.join(OUTPUT_DIR, 'Final_Player_Activation_Data.csv')
        final_ts_dataset = final_ts_dataset.sort_values(['Game_ID', 'Unique_Player_ID', 'True_Game_Elapsed'])
        final_ts_dataset.to_csv(output_path, index=False)

        print(f"\nSUCCESS! Multi-game Time Series saved to {output_path}")

def find_optimal_k(master_ts_df, max_k=10):
    print(f"\nRunning Elbow Method for k=2 to k={max_k}...")

    # Don't forget to include your new feature here!
    features = [
        'Velocity', 'Abs_Acceleration', 'Jerk',
        'Dist_to_Puck', 'Closing_Speed_on_Puck',
        'Dist_to_Centroid', 'Dist_to_Net', 'In_Crease',
        'In_Slot', 'Around_Puck'
    ]

    X = master_ts_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        # n_init='auto' speeds up the process significantly for large datasets
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k={k} calculated. Inertia: {kmeans.inertia_}")

    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method For Optimal k (Micro-States)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.xticks(K_range)
    plt.grid(True)

    output_file = os.path.join(OUTPUT_DIR, 'Elbow_Curve.png')
    plt.savefig(output_file)
    plt.close()

    print(f"\nSaved Elbow Curve to {output_file}")