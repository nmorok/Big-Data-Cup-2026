import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
TRACKING_FILE = os.path.join(BASE_DATA_DIR, "Cluster_Time_Series_Activation/Final_Player_Activation_Data.csv")
EVENT_FILES_PATTERN = os.path.join(BASE_DATA_DIR, "Team*@Team*", "*[Ee]vents*.csv")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_BEFORE = 20  # Extract 20s, but we will strictly plot 5s


def extract_period_number(p_val):
    val_str = str(p_val).upper().strip()
    if val_str in ['OT', 'POT']:
        return 4
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


def create_combined_4panel_plot(trend_df, window_size=5):
    """Plots a 2x2 grid of 5-second trajectories for Goals vs Shots."""
    df = trend_df[trend_df['Rel_Time'] >= -window_size].dropna().copy()
    if df.empty:
        return

    goals = df[df['Event_Type'] == 'Goal']
    shots = df[df['Event_Type'] == 'Shot']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    #fig.suptitle(
    #    f'General Effort vs. Structural Buildup: {window_size}-Second Lead-Up to Event',
    #    fontsize=20, fontweight='bold', y=0.98
    #)

    # ---------------------------------------------------------
    # ROW 1: TOTAL ACTIVATION (Not Significant)
    # ---------------------------------------------------------
    # Top-Left: Attacking Team
    ax1 = axes[0, 0]
    ax1.plot(goals['Rel_Time'], goals['Atk_Active_Smooth'], color='blue', linewidth=4, alpha=0.9, label='Result: GOAL')
    ax1.plot(shots['Rel_Time'], shots['Atk_Active_Smooth'], color='grey', linewidth=2, linestyle='--', alpha=0.8, label='Result: SHOT (No Goal)')
    ax1.set_title("Attacking Team Total Activation\n(Not Statistically Significant)", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Active Players', fontsize=12)
    ax1.axvline(0, color='black', linestyle=':', linewidth=2)
    ax1.set_xlim(-window_size, 0)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(alpha=0.4, linestyle='--')

    # Top-Right: Defending Team
    ax2 = axes[0, 1]
    ax2.plot(goals['Rel_Time'], goals['Def_Active_Smooth'], color='red', linewidth=4, alpha=0.9, label='Result: GOAL')
    ax2.plot(shots['Rel_Time'], shots['Def_Active_Smooth'], color='grey', linewidth=2, linestyle='--', alpha=0.8, label='Result: SHOT (No Goal)')
    ax2.set_title("Defending Team Total Activation\n(Not Statistically Significant)", fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Active Players', fontsize=12)
    ax2.axvline(0, color='black', linestyle=':', linewidth=2)
    ax2.set_xlim(-window_size, 0)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(alpha=0.4, linestyle='--')

    # ---------------------------------------------------------
    # ROW 2: SPECIFIC STRUCTURAL STATES
    # ---------------------------------------------------------
    # Bottom-Left: High Slot (C0)
    ax3 = axes[1, 0]
    ax3.plot(goals['Rel_Time'], goals['C0_Smooth'], color='#FF0000', linewidth=4, alpha=0.9, label='Result: GOAL')
    ax3.plot(shots['Rel_Time'], shots['C0_Smooth'], color='grey', linewidth=2, linestyle='--', alpha=0.8, label='Result: SHOT (No Goal)')
    ax3.set_title("High Slot Profiles (C0)\n(Significant peak: p < 0.05 across all windows)", fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time to Event (Seconds)', fontsize=12)
    ax3.set_ylabel('Average Attacking Players in State', fontsize=12)
    ax3.axvline(0, color='black', linestyle=':', linewidth=2)
    ax3.set_xlim(-window_size, 0)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(alpha=0.4, linestyle='--')

    # Bottom-Right: Deep Crease (C5)
    ax4 = axes[1, 1]
    ax4.plot(goals['Rel_Time'], goals['C5_Smooth'], color='#00BFFF', linewidth=4, alpha=0.9, label='Result: GOAL')
    ax4.plot(shots['Rel_Time'], shots['C5_Smooth'], color='grey', linewidth=2, linestyle='--', alpha=0.8, label='Result: SHOT (No Goal)')
    ax4.set_title("Deep Crease Profiles (C5)\n(Significant at 1s mean: p = 0.008)", fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time to Event (Seconds)', fontsize=12)
    ax4.set_ylabel('Average Attacking Players in State', fontsize=12)
    ax4.axvline(0, color='black', linestyle=':', linewidth=2)
    ax4.set_xlim(-window_size, 0)
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(alpha=0.4, linestyle='--')

    panel_labels = ['A)', 'B)', 'C)', 'D)']
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(-0.08, 1.08, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plot_path = os.path.join(OUTPUT_DIR, f"Combined_4Panel_Trajectories_{window_size}s.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f" -> Generated {window_size}-second 4-Panel Divergence Plot.")


def run_unified_trajectories():
    print("Loading Master Tracking Data...")
    master_df = pd.read_csv(TRACKING_FILE, low_memory=False)
    master_df['Period'] = master_df['Period'].apply(extract_period_number)

    if 'True_Game_Elapsed' in master_df.columns:
        master_df['elapsed_seconds'] = master_df['True_Game_Elapsed']
        master_df = master_df.dropna(subset=['elapsed_seconds'])
    else:
        print(" [WARNING] True_Game_Elapsed missing.")
        return

    # --- DEFINE ALL FLAGS ---
    master_df['Is_Active'] = master_df['State_Cluster'].isin([0, 1, 3, 5]).astype(int)
    master_df['Is_C0_Slot'] = (master_df['State_Cluster'] == 0).astype(int)
    master_df['Is_C3_Possess'] = (master_df['State_Cluster'] == 3).astype(int)
    master_df['Is_C5_Crease'] = (master_df['State_Cluster'] == 5).astype(int)

    event_files = glob.glob(EVENT_FILES_PATTERN)
    print(f"Found {len(event_files)} event files. Extracting Unified Trajectories...")

    trajectory_data = []

    for event_path in event_files:
        folder_name = os.path.basename(os.path.dirname(event_path))
        events_df = pd.read_csv(event_path)
        clock_col = 'Clock' if 'Clock' in events_df.columns else 'Game Clock'
        if clock_col not in events_df.columns:
            continue

        events_df['Period'] = events_df['Period'].apply(extract_period_number)
        events_df['elapsed_seconds'] = events_df.apply(
            lambda r: clock_to_elapsed(r[clock_col], r['Period']), axis=1
        )

        folder_match = re.search(r'Team[\.\s]*([A-Z])\s*@\s*Team[\.\s]*([A-Z])', folder_name, re.IGNORECASE)
        if not folder_match:
            continue

        away_team = f"Team {folder_match.group(1).upper()}"
        home_team = f"Team {folder_match.group(2).upper()}"

        mask = master_df['Game_ID'].str.contains(away_team) & master_df['Game_ID'].str.contains(home_team)
        game_tracking = master_df[mask].copy()
        if game_tracking.empty:
            continue

        # Group by all features
        heartbeat = game_tracking.groupby(['elapsed_seconds', 'Actual_Team'])[
            ['Is_Active', 'Is_C0_Slot', 'Is_C3_Possess', 'Is_C5_Crease']
        ].sum().unstack().fillna(0)

        target_events = events_df[events_df['Event'].str.contains('Goal|Shot', case=False, na=False)]

        for idx, event in target_events.iterrows():
            t = event['elapsed_seconds']
            if pd.isna(t):
                continue

            event_type = 'Goal' if 'Goal' in str(event['Event']).title() else 'Shot'

            atk_raw = str(event['Team']).strip()
            team_letter = re.search(r'([A-Z])$', atk_raw)
            atk_team = f"Team {team_letter.group(1).upper()}" if team_letter else atk_raw

            if atk_team == home_team:
                def_team = away_team
            elif atk_team == away_team:
                def_team = home_team
            else:
                teams_on_ice = [col for col in heartbeat['Is_Active'].columns if 'Team' in col]
                def_team = next((team for team in teams_on_ice if team != atk_team), None)

            if atk_team in heartbeat['Is_Active'].columns and def_team in heartbeat['Is_Active'].columns:
                window = heartbeat[(heartbeat.index >= t - WINDOW_BEFORE) & (heartbeat.index <= t)].copy()

                if not window.empty:
                    for time_index, row in window.iterrows():
                        rel_time = np.round(time_index - t, 1)
                        trajectory_data.append({
                            'Game': folder_name,
                            'Event_ID': f"{folder_name}_{event_type}_{idx}",
                            'Event_Type': event_type,
                            'Rel_Time': rel_time,
                            'Atk_Active': row['Is_Active'][atk_team],
                            'Def_Active': row['Is_Active'][def_team],
                            'Atk_C0_Slot': row['Is_C0_Slot'][atk_team],
                            'Atk_C3_Possess': row['Is_C3_Possess'][atk_team],
                            'Atk_C5_Crease': row['Is_C5_Crease'][atk_team],
                        })

    traj_df = pd.DataFrame(trajectory_data)
    if traj_df.empty:
        print("No trajectory data found.")
        return

    # Group by Event_Type and Rel_Time
    trend_df = traj_df.groupby(['Event_Type', 'Rel_Time'])[
        ['Atk_Active', 'Def_Active', 'Atk_C0_Slot', 'Atk_C3_Possess', 'Atk_C5_Crease']
    ].mean().reset_index()

    # Apply rolling average per event type
    def smooth_group(group):
        group['Atk_Active_Smooth'] = group['Atk_Active'].rolling(3, center=True, min_periods=1).mean()
        group['Def_Active_Smooth'] = group['Def_Active'].rolling(3, center=True, min_periods=1).mean()
        group['C0_Smooth'] = group['Atk_C0_Slot'].rolling(3, center=True, min_periods=1).mean()
        group['C3_Smooth'] = group['Atk_C3_Possess'].rolling(3, center=True, min_periods=1).mean()
        group['C5_Smooth'] = group['Atk_C5_Crease'].rolling(3, center=True, min_periods=1).mean()
        return group

    trend_df = trend_df.groupby('Event_Type', group_keys=False).apply(smooth_group)

    print("\nGenerating Visualizations...")
    create_combined_4panel_plot(trend_df, window_size=5)

    print("\n=== SUCCESS: UNIFIED TRAJECTORY ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    run_unified_trajectories()