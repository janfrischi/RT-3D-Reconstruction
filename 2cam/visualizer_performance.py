import csv
import matplotlib.pyplot as plt
import numpy as np

# Load data from the CSV file
csv_file = "timings.csv"
steps = []
timings = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        step = row[0]
        timing_values = list(map(float, row[1].split(",")))  # Convert to floats
        steps.append(step)
        timings.append(np.mean(timing_values) * 1000)  # Convert seconds to milliseconds

# Calculate percentages
total_time = sum(timings)
percentages = [(timing / total_time) * 100 for timing in timings]

# Create the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(steps, timings, color='skyblue', edgecolor='black')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f} ms", ha="center", va="bottom", fontsize=10)

# Customize the chart
plt.title("Average Timing per Step", fontsize=16)
plt.xlabel("Pipeline Steps", fontsize=14)
plt.ylabel("Time (ms)", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.tight_layout()
plt.savefig("plots/average_timing_per_step.png", dpi=600)
# Show the plot
plt.show()

