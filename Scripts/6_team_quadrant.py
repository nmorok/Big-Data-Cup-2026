import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#OUTPUT_DIR = Adjust this path to your output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_tactical_quadrant():
    # Load in the exact data you provided
    data = {
        'Team': ['Team L', 'Team J', 'Team E', 'Team C', 'Team G', 'Team D',
                 'Team K', 'Team F', 'Team I', 'Team H', 'Team A', 'Team B'],
        'C0_to_C3': [3.83, 3.16, 3.14, 2.92, 2.87, 2.70, 2.58, 2.54, 2.50, 2.38, 2.21, 2.06],
        'C1_to_C3': [3.64, 3.81, 4.53, 3.68, 3.70, 4.05, 4.37, 2.50, 3.87, 4.32, 4.20, 5.80]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(11, 8))

    # Create the scatter plot
    sns.scatterplot(data=df, x='C1_to_C3', y='C0_to_C3', s=200, color='dodgerblue', edgecolor='black', zorder=5)

    # Calculate medians to draw the crosshairs (Quadrant dividers)
    x_mid = df['C1_to_C3'].median()
    y_mid = df['C0_to_C3'].median()

    plt.axvline(x_mid, color='black', linestyle='--', alpha=0.6, zorder=1)
    plt.axhline(y_mid, color='black', linestyle='--', alpha=0.6, zorder=1)

    # Add Team Labels slightly offset from the dots
    for i in range(df.shape[0]):
        plt.text(df['C1_to_C3'].iloc[i] + 0.05,
                 df['C0_to_C3'].iloc[i] + 0.02,
                 df['Team'].iloc[i],
                 fontsize=11, fontweight='500')

    # Add Quadrant Labels to tell the story
   # Quadrant labels pinned to corners
    plt.text(df['C1_to_C3'].max() + 0.3, df['C0_to_C3'].max() + 0.25,
             "Well-Rounded\n(Wins battles & activates slot)",
             color='green', fontsize=12, alpha=0.7, fontweight='bold',
             ha='right', va='top')

    plt.text(df['C1_to_C3'].min() - 0.3, df['C0_to_C3'].max() + 0.25,
             "Precision Assassins\n(Slot specialists, avoid scrums)",
             color='purple', fontsize=12, alpha=0.7, fontweight='bold',
             ha='left', va='top')

    plt.text(df['C1_to_C3'].max() + 0.3, df['C0_to_C3'].min() - 0.25,
             "The Grinders\n(All hustle, poor slot activation)",
             color='darkorange', fontsize=12, alpha=0.7, fontweight='bold',
             ha='right', va='bottom')

    plt.text(df['C1_to_C3'].min() - 0.3, df['C0_to_C3'].min() - 0.25,
             "Static Systems\n(Low recovery, low activation)",
             color='red', fontsize=12, alpha=0.7, fontweight='bold',
             ha='left', va='bottom')

    #plt.title("Team Tactical Fingerprints: How Teams Acquire Possession", fontsize=16, fontweight='bold')
    plt.xlabel("Explosive Recovery: Battle to Possession (C1 $\\rightarrow$ C3 %)", fontsize=13)
    plt.ylabel("High-Value Acquisition: High Slot to Possession (C0 $\\rightarrow$ C3 %)", fontsize=13)

    # Add a bit of padding to the axes so labels don't get cut off
    plt.xlim(df['C1_to_C3'].min() - 0.4, df['C1_to_C3'].max() + 0.5)
    plt.ylim(df['C0_to_C3'].min() - 0.3, df['C0_to_C3'].max() + 0.3)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "Tactical_Quadrant_Scatter.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

create_tactical_quadrant()