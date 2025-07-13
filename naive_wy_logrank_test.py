from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

def logrank_stat(group_1_data, group_0_data, full_dataset):
    """
    Compute the logrank test statistic for two groups.
    """
    if group_0_data.empty or group_1_data.empty:
        return 0.0 # no significance, hence test statistic is zero

    failure_times = sorted(full_dataset.loc[full_dataset["Event"] == 1, "Time"].unique())

    table = []
    for tj in failure_times:
        d0 = ((group_0_data["Time"] == tj) & (group_0_data["Event"] == 1)).sum() # observed events in group 0 at time tj
        d1 = ((group_1_data["Time"] == tj) & (group_1_data["Event"] == 1)).sum() # observed events in group 1 at time tj
        d_total = d0 + d1 # total observed events at time tj

        n1_at_risk = (group_1_data["Time"] >= tj).sum() # patients of group 1 at risk at time tj
        n_at_risk = (full_dataset["Time"] >= tj).sum() # total patients at tisk at time tj

        O1 = d1 # observed events in group 1 at time tj
        E1 = (n1_at_risk / n_at_risk) * d_total if n_at_risk > 0 else 0 # expected events at time tj

        # variance
        if n_at_risk > 1:
            Vj = (n1_at_risk * (n_at_risk - n1_at_risk) * d_total * (n_at_risk - d_total)) / (n_at_risk**2 * (n_at_risk - 1))
        else:
            Vj = 0
            
        table.append({"O1": O1, "E1": E1, "Vj": Vj})

    O1_total = sum(row["O1"] for row in table) # total observed event in group 1
    E1_total = sum(row["E1"] for row in table) # total expected event in group 1
    Vj_total = sum(row["Vj"] for row in table) # total variance

    if Vj_total == 0:
        return 0.0

    z_stat = (O1_total - E1_total)**2 / Vj_total
    
    return z_stat


def naive_wy_logrank_test(data, jp=1000, alpha=0.05):
    """
    A naive implementation of the WY permutations in survival analysis. No bounds or pruning are used.
    """
    original_stat = []
    marker_columns = [col for col in data.columns if col not in ["SampleID", "Event", "Time"]]

    # --- SETUP FOR WESTFALL-YOUNG ---
    print(f"Initializing WY with jp = {jp} permutations and alpha = {alpha}")
    
    # Separate the markers (fixed) from the survival data (to be permuted)
    markers_df = data.drop(columns=["Event", "Time"])
    original_survival_df = data[["Time", "Event"]].copy()
    
    print("1. Generating the permuted datasets...")
    permuted_dfs = [] # list for the jp permuted dataset
    for _ in range(jp):
        permuted_indices = np.random.permutation(original_survival_df.index) # permute the indexes of the dataset
        permuted_survival_df = original_survival_df.loc[permuted_indices].reset_index(drop=True) # reorder rows according to the permutation and reset indexes
        permuted_df = pd.concat([markers_df, permuted_survival_df], axis=1) # concatenate permuted survival data to original markers data
        permuted_dfs.append(permuted_df)
    
    # Array for the z_stats_max of each permutation
    z_max_permutations = np.zeros(jp)
    
    # Marker combinations
    marker_combos = []
    #for r in range(1, len(marker_columns) + 1):
    for r in range(1, 7):
        marker_combos.extend(combinations(marker_columns, r))
        
    # --- Main loop ---
    print("2. Analysis of all combinations on original and permuted data...")
    for marker_combo in tqdm(marker_combos, desc="Analyzing combinations"):
        combo_name = "&".join(marker_combo)
        
        group_1_mask = (data[list(marker_combo)] == 1).all(axis=1) # group 1 mask
        
        group_1_original = data[group_1_mask] # original patients with all the markers in the combo
        group_0_original = data[~group_1_mask] # original patients without all the markers in the combo

        # A. Computation on the original dataset
        stat_orig = logrank_stat(group_1_original, group_0_original, data)
        original_stat.append((combo_name, stat_orig))
        
        # B. Computation on all the permutation to get the final Z_max for each of them
        for j in range(jp):
            permuted_data = permuted_dfs[j]
            
            # Identify groups of patients in the permuted dataset using group 1 mask
            group_1_perm = permuted_data[group_1_mask]
            group_0_perm = permuted_data[~group_1_mask]
            
            stat_perm = logrank_stat(group_1_perm, group_0_perm, permuted_data)
            
            # Update z_max
            z_max_permutations[j] = max(z_max_permutations[j], stat_perm)

    print("\n3. Computation of the corrected significance threshold...")
    z_stat_star = np.quantile(z_max_permutations, 1 - alpha)
    
    print("\n#### Significant Results ####")
    print(f"Corrected Z-statistic threshold (Z-statistic*): {z_stat_star:.4f} (for FWER < {alpha})")
    print("-" * 60)
    print(f"{'Marker combination':45s}\t{'Z-statistic':>8s}")
    print("-" * 60)
    
    # Ordiniamo i risultati per Z-statistic decrescente
    original_stat.sort(key=lambda x: x[1], reverse=True)
    
    significant_count = 0
    BOLD = "\033[1m"
    RESET = "\033[0m"

    for combo_name, stat in original_stat:
        if stat >= z_stat_star and stat > 0:
            print(f"{BOLD}{combo_name:45s}\t{stat:8.4f}{RESET}")
            significant_count += 1
        else:
            print(f"{combo_name:45s}\t{stat:8.4f}")

            
    if significant_count == 0:
        print("No marker combination was found to be significant after correction.")
    else:
        print(f"Found {significant_count} significant combinations after correction")
    print("-" * 60)

# --- MAIN ---

if __name__ == "__main__":
    # Data loading
    try:
        markers = pd.read_csv("sample_logrank_item.csv")
        status = pd.read_csv("sample_logrank_status.csv")
        time = pd.read_csv("sample_logrank_time.csv")
    except FileNotFoundError:
        print("Error: csv files not found.")

    # Columns renaming for consintency
    status.columns = ["SampleID", "Event"]
    time.columns = ["SampleID", "Time"]

    # Data merge
    data = markers.merge(status, on="SampleID").merge(time, on="SampleID")

    # Kaplan-Meier plot for all the patients. Set to 1 if needed.
    plot = 0
    if(plot):
        status = data["Event"]
        time = data["Time"]
        kmf = KaplanMeierFitter()
        kmf.fit(time, status)
        kmf.plot_survival_function()
        plt.show()

    naive_wy_logrank_test(data, jp=1000, alpha=0.05)