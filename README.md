# Analysis of Algorithms Project

## Overview

This project implements and analyzes two fundamental algorithmic paradigms with equal weightage:

1. **Greedy Algorithms**: Applied to optimal crop allocation across agricultural parcels
2. **Divide & Conquer**: Applied to time series data imputation with missing values

Both problems demonstrate the theoretical foundations and practical applications of these algorithmic approaches in solving real-world optimization and data processing challenges.

## Problem Statements

### Problem 1: Greedy Crop Allocation

Given:
- `n` agricultural parcels
- `m` possible crops
- Budget constraint `B`
- For each parcel-crop combination: yield, price, cost, risk factor, and area

**Objective**: Maximize total profit while staying within the budget constraint.

The profit utility function is defined as:
```
U(i,j) = (yield_ij × price_ij) / (cost_ij × risk_ij)
```

### Problem 2: Divide & Conquer Time Series Imputation

Given:
- Time series data with missing values (NaN entries)
- Threshold parameter for determining segment splitting
- Missing data patterns that may be clustered or scattered

**Objective**: Efficiently impute missing values using a divide and conquer approach that:
- Recursively splits the series based on missingness ratio differences
- Applies local linear interpolation within each segment
- Maintains O(n log n) time complexity

## Algorithms Implemented

### Greedy Algorithms (Problem 1)

#### 1. Fractional Greedy Algorithm
- **Strategy**: Sort all parcel-crop combinations by utility ratio (U) in descending order
- **Allocation**: Greedily allocate fractional portions based on available budget and remaining area
- **Time Complexity**: O(nm log(nm))
- **Advantage**: Can utilize partial allocations for better resource utilization

#### 2. Discrete Greedy Algorithm
- **Strategy**: For each parcel, select the crop with highest utility ratio that fits within budget
- **Allocation**: Binary decision (all or nothing) for each parcel
- **Time Complexity**: O(nm)
- **Advantage**: More realistic as it doesn't allow fractional crop allocations

#### 3. Integer Linear Programming (ILP) Optimal Solution
- **Purpose**: Provides optimal baseline for small instances (≤15 parcels)
- **Implementation**: Uses PuLP library with CBC solver
- **Constraint**: Only feasible for small problem sizes due to computational complexity

### Divide & Conquer Algorithm (Problem 2)

#### Time Series Imputation with Recursive Segmentation
- **Strategy**: Recursively divide the time series based on missingness ratio differences
- **Splitting Criterion**: Split when |left_ratio - right_ratio| > threshold
- **Base Case**: Segments with ≤4 elements are not split further
- **Imputation Method**: Local linear interpolation within each segment
- **Time Complexity**: O(n log n) where n is the series length
- **Space Complexity**: O(log n) for recursion stack
- **Advantages**: 
  - Adapts to local data characteristics
  - Handles structured missing patterns efficiently
  - Scalable to large time series

## Files Description

### Greedy Algorithms (Problem 1)
- **`greedy-crop-allocation.py`**: Main implementation file containing all greedy algorithms and experiments
- **`runtime_results.csv`**: Performance comparison data for different problem sizes
- **`profit_comparison_results.csv`**: Profit comparison between greedy and optimal solutions
- **`runtime_plot.png`**: Visualization of algorithm runtime scaling (generated when script runs)
- **`profit_comparison.png`**: Comparison of greedy vs optimal profits (generated when script runs)

### Divide & Conquer (Problem 2)
- **`divideandconquer.py`**: Implementation of divide and conquer time series imputation algorithm
- **Generates**: Runtime complexity visualization and imputation results plots

## Dependencies

```bash
pip install numpy pandas matplotlib pulp
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Problem 1: Running Greedy Crop Allocation Analysis
```bash
python greedy-crop-allocation.py
```

This will:
1. Run runtime experiments across various problem sizes
2. Compare greedy solutions with optimal ILP solutions
3. Generate visualizations and CSV result files

#### Using Individual Greedy Functions

```python
# Generate synthetic dataset
df = generate_synthetic_data(n_parcels=100, m_crops=5, seed=42)

# Run fractional greedy algorithm
fractional_profit = fractional_greedy(df, budget=10000)

# Run discrete greedy algorithm
discrete_profit = discrete_greedy(df, budget=10000)

# Get optimal solution (for small instances)
optimal_profit = ilp_optimal(df, budget=1000, n_limit=15)
```

### Problem 2: Running Divide & Conquer Time Series Imputation
```bash
python divideandconquer.py
```

This will:
1. Demonstrate imputation on synthetic time series with missing data
2. Run runtime complexity experiments
3. Generate visualizations comparing actual vs theoretical O(n log n) performance

#### Using the Divide & Conquer Imputation Function

```python
import numpy as np

# Create time series with missing values
series = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])

# Apply divide and conquer imputation
imputed_series, segments = divide_and_conquer_impute(series, threshold=0.2)
```

## Experimental Results

### Problem 1: Greedy Algorithms Performance

#### Runtime Performance
The experiments test problem sizes ranging from:
- **Parcels**: 100 to 2,000
- **Crops**: 3 to 10
- **Total combinations**: 300 to 20,000

**Key Findings**:
- Fractional greedy shows better scalability with larger problem sizes
- Discrete greedy has higher constant factors but similar asymptotic behavior
- Both algorithms handle large-scale problems efficiently

#### Profit Comparison
Testing on small instances (5-15 parcels) comparing greedy vs optimal:

| Parcels | Greedy Profit | Optimal Profit | Gap |
|---------|---------------|----------------|-----|
| 5       | 101.76        | 108.79         | ~6.5% |
| 10      | 191.46        | 205.35         | ~6.8% |
| 15      | 291.32        | 313.90         | ~7.2% |

**Key Findings**:
- Greedy algorithms achieve approximately 93% of optimal profit
- Performance gap remains relatively stable as problem size increases
- Trade-off between solution quality and computational efficiency

### Problem 2: Divide & Conquer Performance

#### Runtime Complexity Verification
The experiments test time series lengths from 200 to 50,000 elements:

**Key Findings**:
- Measured runtime closely follows theoretical O(n log n) complexity
- Algorithm scales efficiently to large time series (50,000+ elements)
- Recursive segmentation adapts well to different missing data patterns
- Local interpolation within segments provides high-quality imputation

#### Imputation Quality
- Successfully handles structured missing data patterns
- Maintains data continuity across segment boundaries
- Adapts interpolation strategy based on local data characteristics

## Algorithm Analysis

### Problem 1: Greedy Algorithms

#### Time Complexity
- **Fractional Greedy**: O(nm log(nm)) due to sorting step
- **Discrete Greedy**: O(nm) with nested loops over parcels and crops
- **ILP Optimal**: Exponential in worst case, practical only for small instances

#### Space Complexity
- All algorithms: O(nm) for storing the input data structure

#### Approximation Quality
The discrete greedy algorithm provides a good approximation to the optimal solution while being computationally efficient for large-scale problems.

### Problem 2: Divide & Conquer Algorithm

#### Time Complexity
- **Divide & Conquer Imputation**: O(n log n)
  - Recursive splitting: O(log n) levels
  - Linear work at each level: O(n)
  - Total: O(n log n)

#### Space Complexity
- **Recursion Stack**: O(log n) for the recursive calls
- **Data Storage**: O(n) for the input series

#### Algorithm Efficiency
- Adapts to local data patterns through recursive segmentation
- Efficient for large time series while maintaining high imputation quality
- Threshold parameter allows tuning between granularity and performance

## Data Generation and Testing

### Problem 1: Synthetic Agricultural Data
The `generate_synthetic_data()` function creates realistic test cases with:
- **Yield**: 2-6 tons per hectare
- **Price**: $200-500 per ton
- **Cost**: $50-150 per hectare
- **Risk multiplier**: 1.0-1.5
- **Area**: 0.8-1.2 hectares per parcel

Parameters are randomly generated with uniform distributions to ensure diverse and challenging test scenarios.

### Problem 2: Synthetic Time Series Data
The divide and conquer experiments use:
- **Base Signal**: Sinusoidal wave with noise
- **Missing Data Patterns**: Structured gaps at specific intervals
- **Series Lengths**: Range from 200 to 50,000 data points
- **Missing Ratios**: Varied to test different imputation scenarios

## Future Enhancements

### Problem 1: Greedy Algorithms
1. **Advanced Heuristics**: Implement more sophisticated approximation algorithms
2. **Multi-objective Optimization**: Consider environmental impact alongside profit
3. **Dynamic Programming**: Explore DP approaches for specific problem variants
4. **Machine Learning**: Use ML to predict optimal crop selections
5. **Real-world Data**: Integrate with actual agricultural datasets

### Problem 2: Divide & Conquer
1. **Adaptive Thresholding**: Dynamic threshold selection based on data characteristics
2. **Multi-dimensional Time Series**: Extend to handle multivariate time series data
3. **Alternative Imputation Methods**: Implement different local imputation strategies
4. **Parallel Processing**: Leverage the divide and conquer structure for parallelization
5. **Real-world Applications**: Apply to financial, sensor, or environmental time series data

## Authors
Prathmesh Santosh Choudhari and Andrew Rippy

## Purpose

This project demonstrates the implementation and analysis of two fundamental algorithmic paradigms as part of the Analysis of Algorithms coursework:

1. **Greedy Algorithms**: Applied to the crop allocation optimization problem, showcasing greedy choice property and optimal substructure
2. **Divide & Conquer**: Applied to time series imputation, demonstrating recursive problem decomposition and efficient solutions

Both problems have equal weightage and collectively provide comprehensive coverage of algorithmic design techniques, complexity analysis, and practical applications.

## Course Information

**Course**: Analysis of Algorithms  
**Instructor**: Prof. Alin Dobra  
**Purpose**: Educational project demonstrating algorithmic paradigms and their applications