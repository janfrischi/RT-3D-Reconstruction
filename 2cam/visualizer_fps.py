import csv
import numpy as np
import matplotlib.pyplot as plt

# Read the FPS log file
fps_log_file = "fps_log.csv"
timestamps = []
fps_values = []

with open(fps_log_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        timestamps.append(float(row["Timestamp"]))
        fps_values.append(float(row["FPS"]))

# Normalize timestamps to start from 0 for better visualization
timestamps = [t - timestamps[0] for t in timestamps]

# Plot the FPS values over time
plt.figure(figsize=(12, 6))  # Larger figure for better readability

# Main plot
plt.plot(timestamps, fps_values, label="FPS", color='blue', linewidth=1.5)

# Add grid for better readability
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add a horizontal line showing the average FPS
average_fps = np.mean(fps_values)
plt.axhline(y=average_fps, color='red', linestyle='--', linewidth=1, label=f"Average FPS: {average_fps:.2f}")

# Add dynamic axis limits based on data range
plt.xlim(left=min(timestamps), right=max(timestamps))
plt.ylim(bottom=min(fps_values), top=max(fps_values) + 10)

# Add labels and title with more descriptive information
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("FPS", fontsize=12)
plt.title("FPS Over Time", fontsize=14, fontweight='bold')

# Add legend with a background box
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

# Highlight key points (optional: peaks and valleys)
min_idx = np.argmin(fps_values)
max_idx = np.argmax(fps_values)
plt.scatter(timestamps[min_idx], fps_values[min_idx], color='orange', label="Lowest FPS", zorder=5)
plt.scatter(timestamps[max_idx], fps_values[max_idx], color='green', label="Highest FPS", zorder=5)

# Add annotations for peaks and valleys
plt.annotate(f"Min FPS: {fps_values[min_idx]:.2f}",
             (timestamps[min_idx], fps_values[min_idx]),
             textcoords="offset points", xytext=(-15, -20), ha='center', fontsize=10, color='orange')
plt.annotate(f"Max FPS: {fps_values[max_idx]:.2f}",
             (timestamps[max_idx], fps_values[max_idx]),
             textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=10, color='green')

# Save the plot as an image
plt.savefig("plots/fps_over_time.png")
# Show the final plot
plt.show()
