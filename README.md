# Tactical Player Monitoring: Quantifying Team Cohesion Through Behavioral Clustering and Markov State Transitions

**Stathletes Big Data Cup 2026 Submission**  
Cole Morokhovich · Dr. James Watson · Dr. Andrew Berdahl

---

## Overview

In modern hockey analytics, evaluating individual player performance is well-documented, but mathematically quantifying *team cohesion* remains an open challenge. This project asks: **Can behavioral clustering and Markov state transitions measure the dynamic team structures that lead to goals?**

We develop an inferential framework to analyze spatial and kinetic team dynamics from 30 Hz optical tracking data. Players are categorized into discrete behavioral micro-states using K-Means clustering on a 10-dimensional feature vector. We then test whether the composition of those states in the seconds before a shot is statistically different for goals vs. non-goal shots, and model team-level tactical tendencies as Markov chains over those states.

**Key findings:**
- General "activation" (effort) is **not** statistically different between goals and non-goal shots
- Goals are preceded by a significantly higher concentration of attackers in the **High Slot** state (Cluster 0) across all time windows (p < 0.05)
- The **Puck Possession** state (Cluster 3) emerges as a secondary significant differentiator in the critical 5-second lead-up (p = 0.038)
- Markov chain analysis reveals distinct team tactical fingerprints in how they transition from battle states into scoring positions

---

## Pipeline

The analysis runs as a sequential 6-step pipeline. Each script reads from and writes to a shared data directory.

```
1_process_data.py               # Feature engineering + K-Means clustering
       ↓
2_cluster_table.py              # Team-level macro tactical composition table
       ↓
3_preprocess_event_statistics.py  # Multi-window tactical alignment per event
       ↓
4_event_statistics.py           # Welch's t-tests: goals vs. shots
       ↓
5_global_tpm.py                 # Markov Chain transition matrices
       ↓
6_team_quadrant.py              # Team tactical fingerprint quadrant plot
```

`event_buildup.py` is a standalone script that generates the 4-panel trajectory divergence figures shown in the report.

`Report.ipynb` is the complete self-contained Colab notebook with narrative, all code, and embedded outputs as submitted.

---

## Data Directory Structure

The scripts expect the following structure under `BASE_DATA_DIR`:

```
Data/
├── Team.A@Team.B/
│   ├── *Tracking_P1*.csv       # 30Hz optical tracking, Period 1
│   ├── *Tracking_P2*.csv
│   ├── *Tracking_P3*.csv
│   ├── *Tracking_POT*.csv      # Overtime (optional)
│   └── *Events*.csv            # Game event log (shots, goals, etc.)
├── Team.C@Team.D/
│   └── ...
└── Cluster_Time_Series_Activation/   # Created by script 1
    └── Final_Player_Activation_Data.csv
```

**Expected tracking CSV columns:**
`Image Id`, `Period`, `Game Clock` (or `Clock`), `Rink Location X (Feet)`, `Rink Location Y (Feet)`, `Player or Puck`, `Team`, `Player Jersey Number`

**Expected events CSV columns:**
`Period`, `Clock` (or `Game Clock`), `Event`, `Team`

---

## Setup & Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure the data path**

Set `BASE_DATA_DIR` at the top of each script to point to your local data directory. All output directories are created automatically.

**3. Run the pipeline in order**
```bash
python 1_process_data.py
python 2_cluster_table.py
python 3_preprocess_event_statistics.py
python 4_event_statistics.py
python 5_global_tpm.py
python 6_team_quadrant.py

# Optional: trajectory divergence figures
python event_buildup.py
```

**Google Colab users:** All scripts are consolidated in `Report.ipynb`. Mount your Google Drive, set `BASE_DATA_DIR` to your Drive path, and run cells sequentially.

---

## Methods Summary

### 1. Feature Engineering (`1_process_data.py`)

Raw 30 Hz tracking data is processed into a 10-dimensional player-state vector, downsampled to **2 Hz (0.5s intervals)**:

| Feature | Description |
|---|---|
| `Velocity` | Smoothed speed (ft/s), capped at 38.28 ft/s |
| `Abs_Acceleration` | Smoothed absolute acceleration, capped at 16.40 ft/s² |
| `Jerk` | Derivative of acceleration (smoothed) |
| `Dist_to_Puck` | Distance to puck location |
| `Closing_Speed_on_Puck` | Negative derivative of puck distance (positive = approaching) |
| `Dist_to_Centroid` | Distance to own team's spatial centroid |
| `Dist_to_Net` | Minimum distance to either net at (±89, 0) |
| `In_Crease` | Binary: within 8 ft of nearest net |
| `In_Slot` | Binary: within 20 ft of nearest net |
| `Around_Puck` | Binary: within 5 ft of puck |

### 2. K-Means Clustering & Activation (`1_process_data.py`)

K-Means (k=6, determined via elbow method) is fit to all player-states across all games simultaneously. Clusters are interpreted as tactical micro-states:

| Cluster | Label | Active? | Description |
|---|---|---|---|
| 0 | High Slot | Yes | ~14 ft from net, 95% In_Slot — the shooter's pocket |
| 1 | Explosive Battle | Yes | Highest acceleration & jerk — board battles, direction changes |
| 2 | Transition/Glide | No | Moderate velocity, moving away from play |
| 3 | Puck Possession | Yes | 82% Around_Puck, ~4 ft from puck — the puck carrier |
| 4 | Perimeter/Floater | No | ~50 ft from puck, weak-side or line changes |
| 5 | Deep Crease | Yes | 87% In_Crease, ~6.7 ft from net — screening or crashing |

**Activation** is defined as a binary flag: a player is "active" if in Cluster 0, 1, 3, or 5.

### 3. Multi-Window Event Alignment (`3_preprocess_event_statistics.py`)

For each shot and goal, we extract the mean cluster composition of both the attacking and defending teams over 1, 2, 5, 10, and 20-second trailing windows. This produces the `Multi_Window_Tactical_Alignment.csv` used for all downstream statistics.

### 4. Statistical Testing (`4_event_statistics.py`)

Welch's t-tests (unequal variance) compare all cluster metrics between the goal and shot cohorts at each time window.

### 5. Markov Chain Analysis (`5_global_tpm.py`)

Player state sequences are treated as a Markov chain. Transition probabilities are computed from valid consecutive state observations (gap ≤ 0.6s, filtering out whistles and shift changes). Both a global transition matrix and team-specific pathway summaries are produced.

---

## Outputs

All outputs are written to `Outputs`:

| File | Description |
|---|---|
| `Micro_State_Profiles.png` | Cluster profile heatmap (standardized feature means) |
| `Elbow_Curve.png` | Elbow method plot for k selection |
| `Team_Overall_Cluster_Composition.csv` | Team time distribution across all 6 clusters |
| `Multi_Window_Tactical_Alignment.csv` | Per-event tactical metrics across all time windows |
| `Activation_Gap_5s_Stats.png` | Boxplot: 5s activation gap, goals vs. shots |
| `Markov_Global_Matrix.png` / `.csv` | 6×6 global transition probability matrix |
| `Markov_Team_Pathways.csv` | Team-level transition pathway comparison |
| `Tactical_Quadrant_Scatter.png` | Team fingerprint quadrant: recovery vs. slot activation |
| `Combined_4Panel_Trajectories_5s.png` | 4-panel trajectory divergence plot |

---

## Requirements

- Python >= 3.10
- See `requirements.txt` for full dependency list

---

## Authors

Cole Morokhovich, Dr. James Watson, Dr. Andrew Berdahl  
Submitted to the **Stathletes Big Data Cup 2026**
