import pandas as pd
import numpy as np
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DATA_DIR = Adjust this path to your data directory
ALIGNMENT_FILE = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports/Multi_Window_Tactical_Alignment.csv")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "Game_Analytic_Reports")

WINDOWS = [20, 10, 5, 2, 1]

def run_multi_window_statistics():
    print("Loading Multi-Window Tactical Data...")
    df = pd.read_csv(ALIGNMENT_FILE)

    df['Event_Type'] = df['Event'].apply(lambda x: 'Goal' if 'Goal' in str(x).title() else 'Shot')

    goals = df[df['Event_Type'] == 'Goal']
    shots = df[df['Event_Type'] == 'Shot']

    print(f"\nSample Sizes -> Goals: {len(goals)} | Shots: {len(shots)}")

    for w in WINDOWS:
        print(f"\n{'='*40}")
        print(f"       {w}-SECOND WINDOW ANALYSIS")
        print(f"{'='*40}")

        # Define the metrics we generated for this specific window
        # Define ALL the metrics we generated for this specific window
        metrics = [
            f'Atk_Active_{w}s', f'Def_Active_{w}s', f'Activation_Gap_{w}s',
            f'Atk_C0_Slot_{w}s', f'Atk_C1_Battle_{w}s', f'Atk_C3_Possess_{w}s', f'Atk_C5_Crease_{w}s',
            f'Def_C0_Slot_{w}s', f'Def_C1_Battle_{w}s', f'Def_C3_Possess_{w}s', f'Def_C5_Crease_{w}s'
        ]

        for metric in metrics:
            if metric not in df.columns: continue

            g_val = goals[metric].dropna()
            s_val = shots[metric].dropna()

            t_stat, p_val = stats.ttest_ind(g_val, s_val, equal_var=False)

            if p_val < 0.05:
                print(f"✅ SIGNIFICANT: {metric}")
                print(f"   Goals Mean: {g_val.mean():.2f} | Shots Mean: {s_val.mean():.2f}")
                print(f"   P-Value:    {p_val:.4f}\n")
            else:
                print(f"❌ Not Sig: {metric} (P-Value: {p_val:.4f})")

    # Generate a plot for the 5-second Activation Gap
    if 'Activation_Gap_5s' in df.columns:
        plt.figure(figsize=(9, 6))
        sns.boxplot(x='Event_Type', y='Activation_Gap_5s', hue='Event_Type', data=df,
                    palette={'Goal': 'gold', 'Shot': 'lightgrey'}, legend=False, width=0.5,
                    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})

        sns.stripplot(x='Event_Type', y='Activation_Gap_5s', hue='Event_Type', data=df,
                      palette={'Goal': 'darkgoldenrod', 'Shot': 'dimgrey'}, legend=False,
                      alpha=0.5, jitter=True, size=5)

        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
        plt.text(0.5, 0.1, "Offense Dominating", color='red', fontsize=10)

        plt.title("The Activation Gap (5s Lead-up): Goals vs. Shots", fontsize=14, fontweight='bold')
        plt.xlabel("Event Outcome", fontsize=12)
        plt.ylabel("Gap (Attacking Active - Defending Active)", fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        plot_path = os.path.join(OUTPUT_DIR, "Activation_Gap_5s_Stats.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSaved gap plot to {plot_path}")

if __name__ == "__main__":
    run_multi_window_statistics()