import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # Import for smoothing

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

# Limit data to 30 seconds
max_duration = 30  # seconds
filtered_indices = [i for i, t in enumerate(timestamps) if t <= max_duration]
timestamps = [timestamps[i] for i in filtered_indices]
fps_values = [fps_values[i] for i in filtered_indices]

# Apply Savitzky-Golay filter for smoothing
window_size = 15  # Choose an odd number for the window size
poly_order = 2    # Polynomial order for smoothing
smoothed_fps = savgol_filter(fps_values, window_size, poly_order)

# Plot the FPS values over time
plt.figure(figsize=(12, 6))  # Larger figure for better readability

# Main plot
plt.plot(timestamps, smoothed_fps, label="FPS", color='blue', linewidth=2)

# Add grid for better readability
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add a horizontal line showing the average FPS
average_fps = np.mean(smoothed_fps)
plt.axhline(y=average_fps, color='red', linestyle='--', linewidth=1, label=f"Average FPS: {average_fps:.2f}")

# Add dynamic axis limits based on data range
plt.xlim(left=0, right=max_duration)  # Fix x-axis range to 30 seconds
plt.ylim(bottom=min(smoothed_fps), top=max(smoothed_fps) + 10)

# Add labels and title with more descriptive information
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("FPS", fontsize=12)
plt.title("FPS Over Time ", fontsize=16, fontweight='bold', pad=20)

# Add legend with a background box
plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

# Highlight key points (peaks and valleys of smoothed data)
min_idx = np.argmin(smoothed_fps)
max_idx = np.argmax(smoothed_fps)
plt.scatter(timestamps[max_idx], smoothed_fps[max_idx], color='green', label="Highest FPS (Smoothed)", zorder=5)

# Add annotations for peaks and valleys
plt.annotate(f"Max FPS: {smoothed_fps[max_idx]:.2f}",
             (timestamps[max_idx], smoothed_fps[max_idx]),
             textcoords="offset points", xytext=(-15, 10), ha='center', fontsize=10, color='green')

# Save the plot as an image
plt.savefig("plots/fps_over_time_smoothed_30s.png", dpi=600)
# Show the final plot
plt.show()
