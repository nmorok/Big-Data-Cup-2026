# describing teams based on their clusters:
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
#TRACKING_FILE = os.path.join(BASE_DATA_DIR, "Cluster_Time_Series_Activation/Final_Player_Activation_Data.csv")
#OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_team_cluster_composition():
    print("Loading Master Tracking Data...")
    df = pd.read_csv(TRACKING_FILE, low_memory=False)

    # Drop rows missing team or cluster data just to be safe
    df = df.dropna(subset=['Actual_Team', 'State_Cluster'])

    # Clean team names so they match cleanly
    df['Actual_Team'] = df['Actual_Team'].str.replace('.', ' ', regex=False).str.strip()

    print("Calculating overall time spent in each tactical state...")

    # 1. Count the number of half-second frames spent in each cluster per team
    cluster_counts = df.groupby(['Actual_Team', 'State_Cluster']).size().unstack(fill_value=0)

    # 2. Convert raw counts to percentages of total time
    team_composition = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

    # 3. Rename columns for a clean table output
    cluster_labels = {
        0: 'C0_Slot_%',
        1: 'C1_Battle_%',
        2: 'C2_Glide_%',
        3: 'C3_Possess_%',
        4: 'C4_Perimeter_%',
        5: 'C5_Crease_%'
    }
    team_composition = team_composition.rename(columns=cluster_labels)
    team_composition = team_composition.round(2)

    # Add a "Total_Active_%" column (sum of 0, 1, 3, 5)
    team_composition['Total_Active_%'] = team_composition[['C0_Slot_%', 'C1_Battle_%', 'C3_Possess_%', 'C5_Crease_%']].sum(axis=1).round(2)

    # Sort by Total Active to see who the highest-tempo teams are
    team_composition = team_composition.sort_values(by='Total_Active_%', ascending=False)

    print("\n=== MACRO TACTICAL COMPOSITION (Baseline System) ===")
    print(team_composition.to_string())

    out_csv = os.path.join(OUTPUT_DIR, "Team_Overall_Cluster_Composition.csv")
    team_composition.to_csv(out_csv)
    print(f"\nSaved composition table to: {out_csv}")

if __name__ == "__main__":
    generate_team_cluster_composition()