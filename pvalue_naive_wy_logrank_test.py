from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2
from tqdm import tqdm

def calculate_logrank_pvalue(group_1_data, group_0_data, full_dataset):
    """
    Compute the p-value of the log-rank test for two groups.
    """
    if group_0_data.empty or group_1_data.empty:
        return 1.0, 0.0 # p-value non significativo, statistica 0

    failure_times = sorted(full_dataset.loc[full_dataset["Event"] == 1, "Time"].unique())

    table = []
    for tj in failure_times:
        d0 = ((group_0_data["Time"] == tj) & (group_0_data["Event"] == 1)).sum()
        d1 = ((group_1_data["Time"] == tj) & (group_1_data["Event"] == 1)).sum()
        d_total = d0 + d1

        n1_at_risk = (group_1_data["Time"] >= tj).sum()
        n_at_risk = (full_dataset["Time"] >= tj).sum()

        E1 = (n1_at_risk / n_at_risk) * d_total if n_at_risk > 0 else 0
        O1 = d1

        if n_at_risk > 1:
            Vj = (n1_at_risk * (n_at_risk - n1_at_risk) * d_total * (n_at_risk - d_total)) / (n_at_risk**2 * (n_at_risk - 1))
        else:
            Vj = 0
            
        table.append({"O1": O1, "E1": E1, "Vj": Vj})

    O1_total = sum(row["O1"] for row in table)
    E1_total = sum(row["E1"] for row in table)
    Vj_total = sum(row["Vj"] for row in table)

    if Vj_total == 0:
        return 1.0, 0.0

    Z_stat = (O1_total - E1_total)**2 / Vj_total
    p_value = 1 - chi2.cdf(Z_stat, df=1)
    
    return p_value, Z_stat

def my_logrank_test_wy(data, jp=1000, alpha=0.05):
    """
    A naive implementation of the WY permutations in survival analysis. No bounds and pruning are used.
    Versione della tua funzione che integra il framework Simple-WY per il controllo del FWER.
    """
    original_results = []
    marker_columns = [col for col in data.columns if col not in ["SampleID", "Event", "Time"]]

    # --- SETUP FOR WESTFALL-YOUNG ---
    print(f"Initializing Naive-WY with jp = {jp} permutations and alpha = {alpha}")
    
    # Separate the markers (fixed) from the survival data (to be permuted)
    markers_df = data.drop(columns=["Event", "Time"])
    original_survival_df = data[["Time", "Event"]].copy() # Senza .copy(), survival_df_original sarebbe una vista su data, quindi modificare uno potrebbe modificare anche l’altro. Con .copy() si crea invece una copia profonda: modifiche su survival_df_original non influenzano data.
    
    # Generate jp permuted dataset
    print("1. Generating the permuted datasets...")
    permuted_dfs = [] # list for the jp permuted dataset
    for _ in range(jp):
        permuted_indices = np.random.permutation(original_survival_df.index) # permute the indexes of the dataset
        permuted_survival_df = original_survival_df.loc[permuted_indices].reset_index(drop=True) # reorder rows according the permutation and reset indexes
        permuted_df = pd.concat([markers_df, permuted_survival_df], axis=1) # concatenate permuted survival data to original markers data
        permuted_dfs.append(permuted_df)
    
    # Array for the p_min of each permutation
    p_min_permutations = np.ones(jp)
    
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
        p_val_orig, stat_orig = calculate_logrank_pvalue(group_1_original, group_0_original, data)
        original_results.append((combo_name, stat_orig, p_val_orig))
        
        # B. Computation on all the permutation to get the final p_min for each of them
        for j in range(jp):
            permuted_data = permuted_dfs[j]

            # Identify groups of patients using group 1 mask
            group_1_perm = permuted_data[group_1_mask]
            group_0_perm = permuted_data[~group_1_mask]
            
            p_val_perm, _ = calculate_logrank_pvalue(group_1_perm, group_0_perm, permuted_data)
            
            # Update p_min
            p_min_permutations[j] = min(p_min_permutations[j], p_val_perm)

    print("\n3. Computation of the corrected significance threshold...")
    delta_star = np.quantile(p_min_permutations, alpha)
    
    print("\n#### Significant Results after Westfall-Young Correction ####")
    print(f"Corrected p-value threshold (delta*): {delta_star:.4e} (for FWER < {alpha})")
    print("-" * 100)
    print(f"{'Combinazione Marcatori':45s}\t{'Stat':>8s}\t{'P-value Originale':>15s}")
    print("-" * 100)
    
    # Sort p-values
    original_results.sort(key=lambda x: x[2])
    
    significant_count = 0
    
    #for combo_name, stat, pval in original_results:
    #    if pval <= delta_star:
    #        print(f"{combo_name:45s}\t{stat:8.4f}\t{pval:15.4e}")
    #        significant_count += 1

    BOLD = "\033[1m"
    RESET = "\033[0m"

    for combo_name, stat, pval in original_results:
        if pval <= delta_star:
            print(f"{BOLD}{combo_name:45s}\t{stat:8.4f}\t{pval:15.4e}{RESET}")
            significant_count += 1
        else:
            print(f"{combo_name:45s}\t{stat:8.4f}\t{pval:15.4e}")

            
    if significant_count == 0:
        print("No marker combination was found to be significant after correction.")
    print("-" * 100)

# --- MAIN ---

if __name__ == "__main__":
    # Data loading
    markers = pd.read_csv("sample_logrank_item.csv")
    status = pd.read_csv("sample_logrank_status.csv")
    time = pd.read_csv("sample_logrank_time.csv")

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

    my_logrank_test_wy(data)
    print()
    #lib_logrank_test(data)
