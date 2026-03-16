import pandas as pd
import numpy as np
import os
import glob
import re

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
TRACKING_FILE = os.path.join(BASE_DATA_DIR, "Cluster_Time_Series_Activation/Final_Player_Activation_Data.csv")
EVENT_FILES_PATTERN = os.path.join(BASE_DATA_DIR, "Team*@Team*", "*[Ee]vents*.csv")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports")

# Define the three time windows to analyze simultaneously
WINDOWS = [20, 10, 5, 2, 1]

def extract_period_number(p_val):
    val_str = str(p_val).upper().strip()
    if val_str in ['OT', 'POT']: return 4
    try:
        match = re.search(r'\d+', val_str)
        return int(match.group()) if match else 1
    except:
        return 1

def clock_to_elapsed(clock_str, period):
    try:
        p = extract_period_number(period)
        parts = str(clock_str).strip().split(':')
        mm, ss = int(parts[0]), int(parts[1])
        if p > 3 and mm < 20:
            return ((p - 1) * 1200) + ((mm * 60 + ss) if mm == 0 and ss == 0 else (300 - (mm * 60 + ss)))
        return ((p - 1) * 1200) + (1200 - (mm * 60 + ss))
    except:
        return np.nan

def run_multi_window_alignment():
    print("Loading Master Tracking Data...")
    master_df = pd.read_csv(TRACKING_FILE, low_memory=False)
    master_df['Period'] = master_df['Period'].apply(extract_period_number)

    if 'True_Game_Elapsed' in master_df.columns:
        master_df['elapsed_seconds'] = master_df['True_Game_Elapsed']
        master_df = master_df.dropna(subset=['elapsed_seconds'])
    else:
        print(" [WARNING] True_Game_Elapsed missing.")
        return

    # Create distinct flags for our clusters and total activation
    master_df['Is_C0_Slot'] = (master_df['State_Cluster'] == 0).astype(int)
    master_df['Is_C1_Battle'] = (master_df['State_Cluster'] == 1).astype(int)
    master_df['Is_C3_Possess'] = (master_df['State_Cluster'] == 3).astype(int)
    master_df['Is_C5_Crease'] = (master_df['State_Cluster'] == 5).astype(int)
    # Ensure Is_Active is present
    master_df['Is_Active'] = master_df['State_Cluster'].isin([0, 1, 3, 5]).astype(int)

    event_files = glob.glob(EVENT_FILES_PATTERN)
    print(f"Found {len(event_files)} event files. Extracting 20s, 10s, and 5s Profiles...")

    all_game_stats = []

    for event_path in event_files:
        folder_name = os.path.basename(os.path.dirname(event_path))
        events_df = pd.read_csv(event_path)
        clock_col = 'Clock' if 'Clock' in events_df.columns else 'Game Clock'
        if clock_col not in events_df.columns: continue

        events_df['Period'] = events_df['Period'].apply(extract_period_number)
        events_df['elapsed_seconds'] = events_df.apply(lambda r: clock_to_elapsed(r[clock_col], r['Period']), axis=1)

        folder_match = re.search(r'Team[\.\s]*([A-Z])\s*@\s*Team[\.\s]*([A-Z])', folder_name, re.IGNORECASE)
        if not folder_match: continue

        away_team = f"Team {folder_match.group(1).upper()}"
        home_team = f"Team {folder_match.group(2).upper()}"

        mask = master_df['Game_ID'].str.contains(away_team) & master_df['Game_ID'].str.contains(home_team)
        game_tracking = master_df[mask].copy()
        if game_tracking.empty: continue

        # Group by our cluster flags AND the total active flag
        agg_cols = ['Is_C0_Slot', 'Is_C1_Battle', 'Is_C3_Possess', 'Is_C5_Crease', 'Is_Active']
        heartbeat = game_tracking.groupby(['elapsed_seconds', 'Actual_Team'])[agg_cols].sum().unstack().fillna(0)

        target_events = events_df[events_df['Event'].str.contains('Goal|Shot', case=False, na=False)]

        for _, event in target_events.iterrows():
            t = event['elapsed_seconds']
            if pd.isna(t): continue

            atk_raw = str(event['Team']).strip()
            team_letter = re.search(r'([A-Z])$', atk_raw)
            atk_team = f"Team {team_letter.group(1).upper()}" if team_letter else atk_raw

            if atk_team == home_team: def_team = away_team
            elif atk_team == away_team: def_team = home_team
            else:
                teams_on_ice = [col for col in heartbeat['Is_Active'].columns if 'Team' in col]
                def_team = next((team for team in teams_on_ice if team != atk_team), None)

            if atk_team in heartbeat['Is_Active'].columns and def_team in heartbeat['Is_Active'].columns:

                # Base record for the event
                record = {
                    'Game': folder_name,
                    'Event': event['Event'],
                    'Attacking_Team': atk_team,
                    'Defending_Team': def_team,
                    'Time_Elapsed': t
                }

                # Loop through 20, 10, and 5 second windows dynamically
                # Loop through 20, 10, and 5 second windows dynamically
                for w in WINDOWS:
                    window = heartbeat[(heartbeat.index >= t - w) & (heartbeat.index <= t)]

                    if not window.empty:
                        # 1. Total Active Players (Mean over the window)
                        atk_mean = window['Is_Active'][atk_team].mean()
                        def_mean = window['Is_Active'][def_team].mean()
                        record[f'Atk_Active_{w}s'] = round(atk_mean, 2)
                        record[f'Def_Active_{w}s'] = round(def_mean, 2)
                        record[f'Activation_Gap_{w}s'] = round(atk_mean - def_mean, 2)

                        # 2. Peak Attacker Cluster Involvement (Max over the window)
                        record[f'Atk_C0_Slot_{w}s'] = window['Is_C0_Slot'][atk_team].mean()
                        record[f'Atk_C1_Battle_{w}s'] = window['Is_C1_Battle'][atk_team].mean()
                        record[f'Atk_C3_Possess_{w}s'] = window['Is_C3_Possess'][atk_team].mean()
                        record[f'Atk_C5_Crease_{w}s'] = window['Is_C5_Crease'][atk_team].mean()

                        # 3. Peak Defender Cluster Involvement (Max over the window)
                        record[f'Def_C0_Slot_{w}s'] = window['Is_C0_Slot'][def_team].mean()
                        record[f'Def_C1_Battle_{w}s'] = window['Is_C1_Battle'][def_team].mean()
                        record[f'Def_C3_Possess_{w}s'] = window['Is_C3_Possess'][def_team].mean()
                        record[f'Def_C5_Crease_{w}s'] = window['Is_C5_Crease'][def_team].mean()

                # Only append if we successfully got data
                if f'Activation_Gap_{WINDOWS[0]}s' in record:
                    all_game_stats.append(record)

    results_df = pd.DataFrame(all_game_stats)
    out_path = os.path.join(OUTPUT_DIR, "Multi_Window_Tactical_Alignment.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\n=== SUCCESS: MULTI-WINDOW PROFILES SAVED TO {out_path} ===")

if __name__ == "__main__":
    run_multi_window_alignment()