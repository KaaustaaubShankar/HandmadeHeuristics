import matplotlib.pyplot as plt
import numpy as np
# Define a standardized triangular membership function
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a <= x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    return 0

# Generate membership functions using the triangular model

# SIZE
def size_small_tri(x):
    return triangular(x, 0, 0, 10)

def size_large_tri(x):
    return triangular(x, 0, 10, 10)

# WEIGHT
def weight_small_tri(x):
    return triangular(x, 0, 0, 100)

def weight_large_tri(x):
    return triangular(x, 0, 100, 100)

# QUALITY
def quality_bad_tri(x):
    return triangular(x, 0, 0, 0.5)

def quality_medium_tri(x):
    return triangular(x, 0, 0.5, 1)

def quality_good_tri(x):
    return triangular(x, 0.5, 1, 1)


# Define the a_fati function to complete the Mamdani inference process
def b_fita():
    # Inputs
    x = 2  # SIZE input
    y = 25  # WEIGHT input

    # Rule activations (using min operator)
    r1 = min(size_small_tri(x), weight_small_tri(y))  # Rule 1 -> Bad
    r2 = min(size_large_tri(x), weight_small_tri(y))  # Rule 2 -> Medium
    r3 = min(size_small_tri(x), weight_large_tri(y))  # Rule 3 -> Medium
    r4 = min(size_large_tri(x), weight_large_tri(y))  # Rule 4 -> Good

    # Define the output fuzzy sets for QUALITY
    quality_x = np.linspace(0, 1, 100)
    quality_bad_y = [min(r1, quality_bad_tri(q)) for q in quality_x]
    quality_medium_y = [min(max(r2, r3), quality_medium_tri(q)) for q in quality_x]
    quality_good_y = [min(r4, quality_good_tri(q)) for q in quality_x]

    # Aggregation (combine all outputs)
    aggregated_quality = np.fmax(np.fmax(quality_bad_y, quality_medium_y), quality_good_y)
    print(aggregated_quality)

    # Defuzzification using the centroid method
    defuzzified_output = np.sum(quality_x * aggregated_quality) / np.sum(aggregated_quality)

    # Plotting the aggregated output for visualization
    plt.figure(figsize=(8, 4))
    plt.plot(quality_x, quality_bad_y, label='Bad (activated)', color='blue')
    plt.plot(quality_x, quality_medium_y, label='Medium (activated)', color='green')
    plt.plot(quality_x, quality_good_y, label='Good (activated)', color='red')
    plt.fill_between(quality_x, 0, aggregated_quality, color='gray', alpha=0.4, label='Aggregated')
    plt.title("Output Membership Functions and Aggregation (QUALITY)")
    plt.xlabel("Quality")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()

    return defuzzified_output

def a_fati():
    # Inputs
    x = 2  # SIZE input
    y = 25  # WEIGHT input

    # Rule activations (using min operator)
    r1 = min(size_small_tri(x), weight_small_tri(y))  # Rule 1 -> Bad
    r2 = min(size_large_tri(x), weight_small_tri(y))  # Rule 2 -> Medium
    r3 = min(size_small_tri(x), weight_large_tri(y))  # Rule 3 -> Medium
    r4 = min(size_large_tri(x), weight_large_tri(y))  # Rule 4 -> Good

    # Define the output fuzzy sets for QUALITY
    quality_x = np.linspace(0, 1, 100)
    aggregated_quality = np.zeros_like(quality_x)

    # Aggregate all rules into a single fuzzy set
    aggregated_quality = np.fmax(
        np.fmax(
            [min(r1, quality_bad_tri(q)) for q in quality_x],
            [min(max(r2, r3), quality_medium_tri(q)) for q in quality_x]
        ),
        [min(r4, quality_good_tri(q)) for q in quality_x]
    )

    # Defuzzification using the centroid method
    defuzzified_output = np.sum(quality_x * aggregated_quality) / np.sum(aggregated_quality)

    # Plotting the aggregated output for visualization
    plt.figure(figsize=(8, 4))
    plt.plot(quality_x, [min(r1, quality_bad_tri(q)) for q in quality_x], label='Bad (activated)', color='blue')
    plt.plot(quality_x, [min(max(r2, r3), quality_medium_tri(q)) for q in quality_x], label='Medium (activated)', color='green')
    plt.plot(quality_x, [min(r4, quality_good_tri(q)) for q in quality_x], label='Good (activated)', color='red')
    plt.fill_between(quality_x, 0, aggregated_quality, color='gray', alpha=0.4, label='Aggregated')
    plt.title("Output Membership Functions and Aggregation (QUALITY) - a-fati")
    plt.xlabel("Quality")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()

    return defuzzified_output

# Run both methods
a_fati_output = a_fati()
b_fita_output = b_fita()

print(a_fati_output, b_fita_output)


