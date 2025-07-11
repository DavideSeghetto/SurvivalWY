from scipy.stats import chi2
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from itertools import combinations

# === Logrank_test of lifelines library ===
def lib_logrank_test(data):
    results = []

    for marker in data.columns:
        if marker in ["SampleID", "Event", "Time"]:
            continue  # Skip metadata

        group_0 = data[data[marker] == 0]
        group_1 = data[data[marker] == 1]

        if group_0.empty or group_1.empty:
            continue

        result = logrank_test(
            durations_A=group_0["Time"],
            durations_B=group_1["Time"],
            event_observed_A=group_0["Event"],
            event_observed_B=group_1["Event"]
        )

        results.append((marker, result.test_statistic, result.p_value))

        # === Kaplan-Meier plot. Set to 1 if you need the plots for each marker.
        plot = 0
        if(plot):
            kmf = KaplanMeierFitter()
            kmf.fit(group_0["Time"], group_0["Event"], label=f"{marker} = 0")
            ax = kmf.plot_survival_function()
            kmf.fit(group_1["Time"], group_1["Event"], label=f"{marker} = 1")
            ax = kmf.plot_survival_function(ax=ax)
            plt.title(f"Survival Function - {marker}")
            plt.grid(True)
            plt.legend()
            plt.show()
    print("#### Results for logrank test of timelines library ####")
    print("Marker\t\tChi2-stat\tP-value")
    for marker, stat, pval in results:
        print(f"{marker:20s}\t{stat:.4f}\t{pval:.4e}")

def my_logrank_test(data):
    results = []
    # List containing information for each time tj
    failure_times = sorted(data.loc[data["Event"] == 1, "Time"].unique())

    for marker in data.columns:
        table = []
        if marker in ["SampleID", "Event", "Time"]:
            continue
        
        group_0 = data[data[marker] == 0] # group withouth the marker
        group_1 = data[data[marker] == 1] # group with the marker

        if group_0.empty or group_1.empty:
            continue

        for tj in failure_times:
            # 1) Total events at time tj (Event==1)
            d0 = ((group_0["Time"] == tj) & (group_0["Event"] == 1)).sum()
            d1 = ((group_1["Time"] == tj) & (group_1["Event"] == 1)).sum()
            d_total = d0 + d1

            # 2) Patients at risk before tj: Time >= tj
            n1_at_risk = (group_1["Time"] >= tj).sum()
            n_at_risk  = (data["Time"] >= tj).sum()

            # 3) Expected value of events in group 1 at time tj
            E1 = (n1_at_risk / n_at_risk) * d_total

            # 4) Observed number of events in group 1 at time tj
            O1 = d1

            # 5) Variance at time tj
            if n_at_risk != 1:
                # Varianza del paper survival lamp sbagliata (?)
                #Vj = (n1_at_risk * (n_at_risk - n1_at_risk - d_total + d1) * d_total * (n_at_risk - d_total)) / (n_at_risk**2 * (n_at_risk - 1))
                Vj = (n1_at_risk * (n_at_risk - n1_at_risk) * d_total * (n_at_risk - d_total)) / (n_at_risk**2 * (n_at_risk - 1))

            else:
                Vj = 1

            table.append({
                "time":       tj,
                "d_total":    d_total,
                "n1_risk":    n1_at_risk,
                "n_risk":     n_at_risk,
                "O1":         O1,
                "E1":         E1,
                "Vj":         Vj
            })

        O1_total = sum(row["O1"] for row in table)
        E1_total = sum(row["E1"] for row in table)
        Vj_total = sum(row["Vj"] for row in table)

        Z_stat = (O1_total - E1_total)**2 / Vj_total
        
        p_value = 1 - chi2.cdf(Z_stat, df=1)
        results.append((marker, Z_stat, p_value))
    print("#### Results for my logrank test ####")
    print("Marker\t\tChi2-stat\tP-value")
    for marker, stat, pval in results:
        print(f"{marker:20s}\t{stat:.4f}\t{pval:.4e}")

from itertools import combinations
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

def lib_logrank_test2(data):
    results = []

    # Lista dei marker binari (escludendo colonne non di marker)
    marker_columns = [col for col in data.columns if col not in ["SampleID", "Event", "Time"]]

    for r in range(1, len(marker_columns)+1):
    #for r in range(1, 3):
        for marker_combo in combinations(marker_columns, r):
            combo_name = "&".join(marker_combo)

            # Gruppo 1: pazienti con tutti i marker della combinazione
            group_1 = data[(data[list(marker_combo)] == 1).all(axis=1)]
            # Gruppo 0: tutti gli altri
            group_0 = data[~(data["SampleID"].isin(group_1["SampleID"]))]

            if group_0.empty or group_1.empty:
                continue

            result = logrank_test(
                durations_A=group_0["Time"],
                durations_B=group_1["Time"],
                event_observed_A=group_0["Event"],
                event_observed_B=group_1["Event"]
            )

            results.append((combo_name, result.test_statistic, result.p_value))

            # === Kaplan-Meier plot. Set to 1 if you need the plots
            plot = 0
            if plot:
                kmf = KaplanMeierFitter()
                kmf.fit(group_0["Time"], group_0["Event"], label=f"{combo_name} = 0")
                ax = kmf.plot_survival_function()
                kmf.fit(group_1["Time"], group_1["Event"], label=f"{combo_name} = 1")
                ax = kmf.plot_survival_function(ax=ax)
                plt.title(f"Survival Function - {combo_name}")
                plt.grid(True)
                plt.legend()
                plt.show()

    print("#### Results for logrank test of lifelines library ####")
    print("Markers Combination\tChi2-stat\tP-value")
    for marker, stat, pval in results:
        print(f"{marker:30s}\t{stat:.4f}\t{pval:.4e}")


def my_logrank_test2(data):
    results = []
    failure_times = sorted(data.loc[data["Event"] == 1, "Time"].unique())

    # Lista dei marker binari (escludendo colonne non di marker)
    marker_columns = [col for col in data.columns if col not in ["SampleID", "Event", "Time"]]

    # Per ogni combinazione non vuota di marker
    for r in range(1, len(marker_columns)+1):
    #for r in range(1, 3):
        for marker_combo in combinations(marker_columns, r):
            combo_name = "&".join(marker_combo)

            # Gruppo 1: pazienti che hanno TUTTI i marker nella combinazione
            group_1 = data[(data[list(marker_combo)] == 1).all(axis=1)]
            # Gruppo 0: tutti gli altri
            group_0 = data[~(data["SampleID"].isin(group_1["SampleID"]))]

            if group_0.empty or group_1.empty:
                continue

            table = []
            for tj in failure_times:
                d0 = ((group_0["Time"] == tj) & (group_0["Event"] == 1)).sum()
                d1 = ((group_1["Time"] == tj) & (group_1["Event"] == 1)).sum()
                d_total = d0 + d1

                n1_at_risk = (group_1["Time"] >= tj).sum()
                n_at_risk = (data["Time"] >= tj).sum()

                E1 = (n1_at_risk / n_at_risk) * d_total if n_at_risk > 0 else 0
                O1 = d1

                if n_at_risk > 1:
                    Vj = (n1_at_risk * (n_at_risk - n1_at_risk) * d_total * (n_at_risk - d_total)) / \
                         (n_at_risk ** 2 * (n_at_risk - 1))
                else:
                    Vj = 1

                table.append({"O1": O1, "E1": E1, "Vj": Vj})

            O1_total = sum(row["O1"] for row in table)
            E1_total = sum(row["E1"] for row in table)
            Vj_total = sum(row["Vj"] for row in table)

            if Vj_total == 0:
                continue

            Z_stat = (O1_total - E1_total) ** 2 / Vj_total
            p_value = 1 - chi2.cdf(Z_stat, df=1)
            results.append((combo_name, Z_stat, p_value))

    # Stampa risultati
    print("#### Results for my logrank test (combinations of markers) ####")
    print("Markers Combination\tChi2-stat\tP-value")
    for marker, stat, pval in results:
        print(f"{marker:45s}\t{stat:.4f}\t{pval:.4e}")


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

    # Esecuzione solo del tuo logrank test personalizzato
    my_logrank_test2(data)
    print()
    lib_logrank_test2(data)