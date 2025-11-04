import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pulp

# ---------- Synthetic Dataset Generator ----------
def generate_synthetic_data(n_parcels=200, m_crops=5, seed=42):
    np.random.seed(seed)
    parcels = np.arange(n_parcels)
    crops = np.arange(m_crops)

    data = []
    for i in parcels:
        for j in crops:
            y = np.random.uniform(2, 6)       # yield per ha
            v = np.random.uniform(200, 500)   # price per ton
            w = np.random.uniform(50, 150)    # resource cost per ha
            r = np.random.uniform(1.0, 1.5)   # risk multiplier
            a = np.random.uniform(0.8, 1.2)   # parcel area
            U = (y * v) / (w * r)
            data.append([i, j, y, v, w, r, a, U])

    df = pd.DataFrame(data, columns=['parcel', 'crop', 'yield', 'price', 'cost', 'risk', 'area', 'U'])
    return df

# ---------- Fractional Greedy Allocation ----------
def fractional_greedy(df, budget=10000):
    df = df.sort_values('U', ascending=False).reset_index(drop=True)
    B = budget
    remaining_area = df.groupby('parcel')['area'].max().to_dict()
    total_value = 0

    for _, row in df.iterrows():
        pid, area, U, cost = row['parcel'], row['area'], row['U'], row['cost']
        if B <= 0 or remaining_area[pid] <= 0:
            continue
        max_area = min(remaining_area[pid], B / cost)
        profit = max_area * (row['yield'] * row['price']) / (row['risk'] * row['cost'])
        B -= max_area * cost
        remaining_area[pid] -= max_area
        total_value += profit
    return total_value

# ---------- Discrete Greedy Allocation ----------
def discrete_greedy(df, budget=10000):
    remaining_B = budget
    parcels = df['parcel'].unique()
    total_value = 0

    for pid in parcels:
        sub = df[df['parcel'] == pid].copy()
        sub['U'] = (sub['yield'] * sub['price']) / (sub['cost'] * sub['risk'])
        sub = sub.sort_values('U', ascending=False)
        for _, row in sub.iterrows():
            if row['cost'] * row['area'] <= remaining_B:
                remaining_B -= row['cost'] * row['area']
                total_value += (row['yield'] * row['price'] * row['area']) / (row['risk'] * row['cost'])
                break
    return total_value

# ---------- ILP Optimal (for small instances) ----------
def ilp_optimal(df, budget=1000, n_limit=15):
    df = df[df['parcel'] < n_limit]
    prob = pulp.LpProblem("CropAllocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', ((i, j) for i, j in zip(df['parcel'], df['crop'])), cat='Binary')
    value = {(i, j): (df[(df.parcel==i)&(df.crop==j)]['yield']*df[(df.parcel==i)&(df.crop==j)]['price'] /
                      (df[(df.parcel==i)&(df.crop==j)]['risk']*df[(df.parcel==i)&(df.crop==j)]['cost'])).values[0]
             for i, j in zip(df['parcel'], df['crop'])}
    cost = {(i, j): df[(df.parcel==i)&(df.crop==j)]['cost'].values[0]*df[(df.parcel==i)&(df.crop==j)]['area'].values[0]
            for i, j in zip(df['parcel'], df['crop'])}

    # Objective
    prob += pulp.lpSum([x[(i,j)] * value[(i,j)] for (i,j) in x])
    # Constraints
    for i in df['parcel'].unique():
        prob += pulp.lpSum([x[(i,j)] for j in df['crop'].unique()]) <= 1
    prob += pulp.lpSum([x[(i,j)] * cost[(i,j)] for (i,j) in x]) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective)

# ---------- Runtime Experiment ----------
def run_runtime_experiment():
    results = []
    for n in [100, 300, 500, 1000, 2000]:
        for m in [3, 5, 10]:
            df = generate_synthetic_data(n, m)
            B = 10000
            t1 = time.perf_counter(); fractional_greedy(df, B); t_frac = time.perf_counter()-t1
            t2 = time.perf_counter(); discrete_greedy(df, B); t_disc = time.perf_counter()-t2
            results.append((n, m, n*m, t_frac, t_disc))
    res = pd.DataFrame(results, columns=['Parcels','Crops','N','Fractional_Time','Discrete_Time'])
    res.to_csv("runtime_results.csv", index=False)
    print("✅ Saved runtime_results.csv")
    
    plt.figure(figsize=(7,5))
    plt.loglog(res['N'], res['Fractional_Time'], marker='o', label='Fractional Greedy')
    plt.loglog(res['N'], res['Discrete_Time'], marker='s', label='Discrete Greedy')
    plt.xlabel('Number of (Parcel × Crop) pairs (N)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison of Greedy Algorithms')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("runtime_plot.png", dpi=300)
    plt.show()

# ---------- Profit Comparison ----------
def run_profit_experiment():
    ns = [5, 10, 15]
    greedy_vals, opt_vals = [], []
    for n in ns:
        df = generate_synthetic_data(n, m_crops=5)
        greedy_vals.append(discrete_greedy(df, 1000))
        opt_vals.append(ilp_optimal(df, 1000, n_limit=n))
    
    result_df = pd.DataFrame({'Parcels': ns, 'Greedy_Profit': greedy_vals, 'ILP_Optimal_Profit': opt_vals})
    result_df.to_csv("profit_comparison_results.csv", index=False)
    print("✅ Saved profit_comparison_results.csv")
    
    plt.figure(figsize=(6,4))
    plt.plot(ns, greedy_vals, marker='o', label='Greedy Profit')
    plt.plot(ns, opt_vals, marker='s', label='ILP Optimal Profit')
    plt.xlabel('Number of Parcels (n)')
    plt.ylabel('Total Profit')
    plt.title('Greedy vs ILP Optimal Profit')
    plt.legend()
    plt.tight_layout()
    plt.savefig("profit_comparison.png", dpi=300)
    plt.show()

# ---------- Run Experiments ----------
if __name__ == "__main__":
    print("Running Runtime Experiment...")
    run_runtime_experiment()

    print("Running Profit Comparison Experiment...")
    run_profit_experiment()

    print("✅ All graphs and CSVs saved successfully:")
    print("   • runtime_plot.png")
    print("   • profit_comparison.png")
    print("   • runtime_results.csv")
    print("   • profit_comparison_results.csv")
