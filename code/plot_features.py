import pandas as pd
import matplotlib.pyplot as plt
import os

fold = "results/patient_features"
files = os.listdir(fold)

for file in files:
    file_path = os.path.join(fold, file)
    print(file)
    # Load CSV with no headers
    data = pd.read_csv(file_path, header=None)

    # Flatten row to a list
    values = data.values.flatten().tolist()

    # Window sizes (same for both stromal and peritumoral)
    window_sizes = [200, 250, 300, 350, 400, 450, 500, 550, 600]

    # Extract values
    stromal_mean = values[0:18:2]
    stromal_max = values[1:18:2]
    peritumoral_mean = values[18:36:2]
    peritumoral_max = values[19:36:2]

    plot_name = file.rsplit('.', 1)[0] + ".png"

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, stromal_mean, marker='o', label='Stromal Mean')
    plt.plot(window_sizes, stromal_max, marker='o', label='Stromal Max')
    plt.plot(window_sizes, peritumoral_mean, marker='s', label='Peritumoral Mean')
    plt.plot(window_sizes, peritumoral_max, marker='s', label='Peritumoral Max')

    plt.xlabel('Window Size')
    plt.ylabel('Feature Value')
    plt.title('Feature Values vs. Window Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{plot_name}")
    plt.close()
