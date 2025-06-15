# Factor Analysis
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of Data:",X_scaled.shape)

#initialize and fit factor analysis
fa = FactorAnalysis(n_components=10,random_state=10)
X_fa = fa.fit_transform(X_scaled)

print("shape after fa:",X_fa.shape)

# factor loadings
loadings = pd.DataFrame(fa.components_.T,columns=[f'Factor_{i+1}' for i in range(10)])

print(loadings.head())

# Visualize factor loadings
plt.figure(figsize=(12, 6))
sns.heatmap(loadings, cmap='coolwarm', center=0)
plt.title("Factor Loadings Heatmap")
plt.xlabel("Factors")
plt.ylabel("Original Features")
plt.show()

#compute communalities

communalities = np.sum(loadings**2,axis=1)

# Plot communalities
plt.figure(figsize=(10,4))
plt.bar(range(len(communalities)), communalities)
plt.title("Communalities (Variance Explained by Factors)")
plt.xlabel("Feature Index")
plt.ylabel("Communality")
plt.show()


# Detailed Communalities plot

plt.figure(figsize=(10, 4))
bars = plt.bar(range(len(communalities)), communalities)
plt.title("Communalities: How Much Variance Each Variable Shares with the Factors")
plt.xlabel("Feature Index")
plt.ylabel("Communality")

# Highlight low-communalities
for idx, bar in enumerate(bars):
    if communalities[idx] < 0.35:
        bar.set_color('red')
        plt.text(idx, communalities[idx] + 0.02, f"{communalities[idx]:.2f}", ha='center', color='red', fontsize=8)
    else:
        plt.text(idx, communalities[idx] + 0.02, f"{communalities[idx]:.2f}", ha='center', color='black', fontsize=8)

plt.axhline(0.35, linestyle='--', color='gray', label="Low Communality Threshold")
plt.legend()
plt.tight_layout()
plt.show()

# Feature index

import matplotlib.pyplot as plt
import numpy as np


communalities = np.array([
    0.06, 0.93, 0.73, 0.71, 0.65, 0.47, 0.45, 0.25,
    0.01, 0.77, 0.74, 0.72, 0.62, 0.38, 0.02, 0.02,
    0.57, 0.55, 0.48, 0.49, 0.42, 0.72, 0.61, 0.51,
    0.59, 0.59, 0.50, 0.53, 0.35, 0.72, 0.64, 0.47,
    0.41, 0.59, 0.56, 0.57, 0.92, 0.64, 0.60, 0.45,
    0.05, 0.00, 0.82, 0.72, 0.63, 0.46, 0.38, 0.36,
    0.93, 0.95, 0.81, 0.64, 0.54, 0.29, 0.10, 0.03,
    0.82, 0.82, 0.72, 0.63, 0.36, 0.01, 0.64, 0.54
])

# Ensure it's reshaped to the original 8x8 image format
#communalities_reshaped = communalities.reshape((8, 8))

# Plot as an 8x8 image
plt.figure(figsize=(6, 6))
plt.imshow(communalities_reshaped, cmap='viridis', interpolation='nearest')
plt.title("🔍 Communality Map of Digits Dataset (8x8 Grid)")
plt.colorbar(label='Communality (Variance Explained)')
plt.xticks(range(8))
plt.yticks(range(8))
plt.gca().invert_yaxis()  # To match image origin at top-left
plt.grid(False)
plt.tight_layout()
plt.show()