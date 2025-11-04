import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------
# Divide & Conquer Imputation
# -------------------------------

def missing_ratio(series, start, end):
    """Compute missing ratio for a given window [start:end)."""
    sub = series[start:end]
    return np.mean(np.isnan(sub)) if len(sub) > 0 else 0

def split_series(series, start, end, threshold, segments):
    """
    Recursively split based on difference in missingness ratio.
    Uses index references (no copying).
    """
    if end - start <= 4:  # too small to split
        segments.append((start, end))
        return

    mid = (start + end) // 2
    left_ratio = missing_ratio(series, start, mid)
    right_ratio = missing_ratio(series, mid, end)

    if abs(left_ratio - right_ratio) > threshold:
        split_series(series, start, mid, threshold, segments)
        split_series(series, mid, end, threshold, segments)
    else:
        segments.append((start, end))

def divide_and_conquer_split(series, threshold=0.2):
    """Wrapper for recursive segmentation."""
    segments = []
    split_series(series, 0, len(series), threshold, segments)
    return segments

def local_impute(segment, global_mean):
    """Local linear interpolation for a numeric array segment."""
    nans = np.isnan(segment)
    if np.all(nans):
        return np.full_like(segment, global_mean)
    x = np.arange(len(segment))
    segment[nans] = np.interp(x[nans], x[~nans], segment[~nans])
    return segment

def divide_and_conquer_impute(series, threshold=0.2):
    """Main D&C imputation algorithm."""
    series = series.copy()
    global_mean = np.nanmean(series)
    segments = divide_and_conquer_split(series, threshold)
    for (start, end) in segments:
        segment = series[start:end]
        imputed_segment = local_impute(segment, global_mean)
        series[start:end] = imputed_segment
    return series, segments

# -------------------------------
# Experiment Setup
# -------------------------------

np.random.seed(0)
t = np.linspace(0, 10, 400)
true_series = np.sin(t) + 0.1 * np.random.randn(len(t))
series_with_gaps = true_series.copy()

# introduce structured missing data
series_with_gaps[80:100] = np.nan
series_with_gaps[200:220] = np.nan
series_with_gaps[350:360] = np.nan

# Run algorithm
start_time = time.time()
imputed, segs = divide_and_conquer_impute(series_with_gaps, threshold=0.15)
runtime = time.time() - start_time

# -------------------------------
# Visualization: Imputation
# -------------------------------

plt.figure(figsize=(10, 5))
plt.plot(t, true_series, label='True Series', alpha=0.6)
plt.plot(t, series_with_gaps, 'o', label='With Missing Data', alpha=0.4)
plt.plot(t, imputed, label='Imputed (D&C)', linewidth=2)
plt.title(f"Divide & Conquer Imputation (Runtime: {runtime:.4f}s)")
plt.legend()
plt.show()

# -------------------------------
# Runtime Complexity Experiment
# -------------------------------

sizes = [200, 400, 800, 1600, 3200, 6400, 9600, 12800, 19200, 25600, 30000, 35000, 40000, 45000,50000]
times = []

for n in sizes:
    t = np.linspace(0, 10, n)
    series = np.sin(t)
    # Add random missing values
    mask = np.random.rand(n) < 0.1
    series[mask] = np.nan

    start = time.time()
    divide_and_conquer_impute(series, threshold=0.15)
    times.append(time.time() - start)

# Theoretical O(n log n) for comparison
x_log_x = np.array([n * np.log2(n) for n in sizes])
x_log_x = (x_log_x / np.max(x_log_x) * np.max(times))

# -------------------------------
# Visualization: Runtime vs n log n
# -------------------------------

plt.figure(figsize=(8, 5))
plt.plot(sizes, times, 'o-', label='Measured Runtime')
plt.plot(sizes, x_log_x, '--', label=r'$O(n \log n)$ (normalized)')
plt.xlabel("Time Series Length (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Input Size — Expected O(n log n)")
plt.legend()
plt.grid(True)
plt.show()