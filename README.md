# Greedy Crop Allocation Algorithm

## Overview

This project implements and analyzes greedy algorithms for optimal crop allocation across agricultural parcels. The problem involves maximizing profit by selecting the best crop for each parcel while considering resource constraints like budget limitations.

## Problem Statement

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

## Algorithms Implemented

### 1. Fractional Greedy Algorithm
- **Strategy**: Sort all parcel-crop combinations by utility ratio (U) in descending order
- **Allocation**: Greedily allocate fractional portions based on available budget and remaining area
- **Time Complexity**: O(nm log(nm))
- **Advantage**: Can utilize partial allocations for better resource utilization

### 2. Discrete Greedy Algorithm
- **Strategy**: For each parcel, select the crop with highest utility ratio that fits within budget
- **Allocation**: Binary decision (all or nothing) for each parcel
- **Time Complexity**: O(nm)
- **Advantage**: More realistic as it doesn't allow fractional crop allocations

### 3. Integer Linear Programming (ILP) Optimal Solution
- **Purpose**: Provides optimal baseline for small instances (≤15 parcels)
- **Implementation**: Uses PuLP library with CBC solver
- **Constraint**: Only feasible for small problem sizes due to computational complexity

## Files Description

- **`greedy-crop-allocation.py`**: Main implementation file containing all algorithms and experiments
- **`runtime_results.csv`**: Performance comparison data for different problem sizes
- **`profit_comparison_results.csv`**: Profit comparison between greedy and optimal solutions
- **`runtime_plot.png`**: Visualization of algorithm runtime scaling (generated when script runs)
- **`profit_comparison.png`**: Comparison of greedy vs optimal profits (generated when script runs)

## Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Analysis
```bash
python greedy-crop-allocation.py
```

This will:
1. Run runtime experiments across various problem sizes
2. Compare greedy solutions with optimal ILP solutions
3. Generate visualizations and CSV result files

### Using Individual Functions

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

## Experimental Results

### Runtime Performance
The experiments test problem sizes ranging from:
- **Parcels**: 100 to 2,000
- **Crops**: 3 to 10
- **Total combinations**: 300 to 20,000

**Key Findings**:
- Fractional greedy shows better scalability with larger problem sizes
- Discrete greedy has higher constant factors but similar asymptotic behavior
- Both algorithms handle large-scale problems efficiently

### Profit Comparison
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

## Algorithm Analysis

### Time Complexity
- **Fractional Greedy**: O(nm log(nm)) due to sorting step
- **Discrete Greedy**: O(nm) with nested loops over parcels and crops
- **ILP Optimal**: Exponential in worst case, practical only for small instances

### Space Complexity
- All algorithms: O(nm) for storing the input data structure

### Approximation Quality
The discrete greedy algorithm provides a good approximation to the optimal solution while being computationally efficient for large-scale problems.

## Synthetic Data Generation

The `generate_synthetic_data()` function creates realistic test cases with:
- **Yield**: 2-6 tons per hectare
- **Price**: $200-500 per ton
- **Cost**: $50-150 per hectare
- **Risk multiplier**: 1.0-1.5
- **Area**: 0.8-1.2 hectares per parcel

Parameters are randomly generated with uniform distributions to ensure diverse and challenging test scenarios.

## Future Enhancements

1. **Advanced Heuristics**: Implement more sophisticated approximation algorithms
2. **Multi-objective Optimization**: Consider environmental impact alongside profit
3. **Dynamic Programming**: Explore DP approaches for specific problem variants
4. **Machine Learning**: Use ML to predict optimal crop selections
5. **Real-world Data**: Integrate with actual agricultural datasets

## Author
Prathmesh Santosh Choudhari and Andrew Rippy

Analysis of Algorithms Course Project - Implementation and comparative study of greedy algorithms for the crop allocation optimization problem.

## Purpose

This project is for educational purposes as part of coursework Analysis of Algorithm by Prof. Alin Dobra.