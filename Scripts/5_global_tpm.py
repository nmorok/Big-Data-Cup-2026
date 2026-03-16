import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
TRACKING_FILE = os.path.join(BASE_DATA_DIR, "Cluster_Time_Series_Activation/Final_Player_Activation_Data.csv")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 14})

# Define our cluster names for clean plotting
CLUSTER_LABELS = [
    '0: High Slot',
    '1: Battle/Scrum',
    '2: Transition/Glide',
    '3: Possession',
    '4: Perimeter/Floater',
    '5: Deep Crease'
]

def build_transition_matrices():
    print("Loading Tracking Data for Markov Chains...")
    df = pd.read_csv(TRACKING_FILE, low_memory=False)

    # Ensure we have the exact chronological ordering required for a Markov Chain
    df = df.sort_values(['Game_ID', 'Unique_Player_ID', 'True_Game_Elapsed']).reset_index(drop=True)

    print("Calculating player state transitions...")
    # Shift the cluster column up by 1 to get the "Next State"
    df['Next_State'] = df.groupby(['Game_ID', 'Unique_Player_ID'])['State_Cluster'].shift(-1)

    # Shift the time column up by 1 to get the exact time gap to the next state
    df['Time_to_Next'] = df.groupby(['Game_ID', 'Unique_Player_ID'])['True_Game_Elapsed'].diff().shift(-1)

    # ---------------------------------------------------------
    # THE WHISTLE FILTER:
    # Since our data is 2Hz (0.5s gaps), we drop any transition
    # where the gap is greater than 0.6s (whistles, shifts, etc.)
    # ---------------------------------------------------------
    valid_transitions = df[(df['Time_to_Next'] > 0) & (df['Time_to_Next'] <= 0.6)].copy()

    # Drop NaNs just in case (the last frame for every player will naturally be NaN)
    valid_transitions = valid_transitions.dropna(subset=['Next_State'])

    # Ensure integer types for the crosstab
    valid_transitions['State_Cluster'] = valid_transitions['State_Cluster'].astype(int)
    valid_transitions['Next_State'] = valid_transitions['Next_State'].astype(int)

    # ==========================================
    # 1. THE GLOBAL MATRIX
    # ==========================================
    print("\nGenerating Global Transition Matrix...")
    global_matrix = pd.crosstab(
        valid_transitions['State_Cluster'],
        valid_transitions['Next_State'],
        normalize='index'
    ) * 100 # Convert to percentage (0-100)

    # --- NEW: Save the Raw 6x6 Global Matrix to CSV ---
    clean_global_matrix = global_matrix.copy()
    clean_global_matrix.index = CLUSTER_LABELS   # Name the rows
    clean_global_matrix.columns = CLUSTER_LABELS # Name the columns

    global_csv_path = os.path.join(OUTPUT_DIR, "Markov_Global_Matrix.csv")
    clean_global_matrix.to_csv(global_csv_path)
    print(f" -> Saved Global Matrix CSV to {global_csv_path}")
    # --------------------------------------------------

    # Plot the Global Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(global_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=CLUSTER_LABELS, yticklabels=CLUSTER_LABELS, 
            vmin=0, vmax=100,
            annot_kws={"size": 14})
    plt.ylabel("Current State ($S_t$)", fontsize=16)
    plt.xlabel("Next State ($S_{t+0.5s}$)", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    
    global_plot_path = os.path.join(OUTPUT_DIR, "Markov_Global_Matrix.png")
    plt.savefig(global_plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f" -> Saved Global Matrix to {global_plot_path}")

    # ==========================================
    # 2. TEAM-SPECIFIC MATRICES
    # ==========================================
    print("\nExtracting Team-Specific Matrices...")
    teams = valid_transitions['Actual_Team'].unique()

    team_data = []

    for team in teams:
        team_df = valid_transitions[valid_transitions['Actual_Team'] == team]

        if team_df.empty: continue

        team_matrix = pd.crosstab(
            team_df['State_Cluster'],
            team_df['Next_State'],
            normalize='index'
        ) * 100

        # Safely extract specific pathways of interest to compare teams
        def get_prob(matrix, start, end):
            try: return round(matrix.loc[start, end], 2)
            except KeyError: return 0.0

        team_data.append({
            'Team': team,
            # The Diagonal: Stickiness
            'Stickiness_C0_Slot': get_prob(team_matrix, 0, 0),
            'Stickiness_C3_Possess': get_prob(team_matrix, 3, 3),

            # The Golden Pathways we identified
            'Pathway_C2_to_C0 (Off-Ball IQ)': get_prob(team_matrix, 2, 0),
            'Pathway_C0_to_C3 (Slot Targeting)': get_prob(team_matrix, 0, 3),
            'Pathway_C1_to_C3 (Puck Recovery)': get_prob(team_matrix, 1, 3)
        })

    # Save the summarized team comparison to CSV
    team_comparison_df = pd.DataFrame(team_data).sort_values('Pathway_C0_to_C3 (Slot Targeting)', ascending=False)
    team_csv_path = os.path.join(OUTPUT_DIR, "Markov_Team_Pathways.csv")
    team_comparison_df.to_csv(team_csv_path, index=False)

    print(f"\n=== SUCCESS: MARKOV CHAIN EXTRACTION COMPLETE ===")
    print("\nTop 5 Teams by Slot Targeting (C0 -> C3):")
    print(team_comparison_df[['Team', 'Pathway_C0_to_C3 (Slot Targeting)']].head(5))

if __name__ == "__main__":
    build_transition_matrices()