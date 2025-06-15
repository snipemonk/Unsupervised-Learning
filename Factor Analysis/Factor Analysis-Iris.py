# implementing factor analysis on Iris Dataset

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=0)
X_fa = fa.fit_transform(X_scaled)

# Create loadings DataFrame
loadings = pd.DataFrame(fa.components_.T,
                        index=iris.feature_names,
                        columns=['Factor1', 'Factor2'])

# Plot factor loadings heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title("Factor Loadings (Iris Dataset)")
plt.tight_layout()
plt.show()

# Create DataFrame for factor scores
fa_scores = pd.DataFrame(X_fa, columns=['Factor1', 'Factor2'])
fa_scores['species'] = y

# Plot factor scores by species
plt.figure(figsize=(6, 5))
sns.scatterplot(data=fa_scores, x='Factor1', y='Factor2', hue='species', palette='Set1')
plt.title("Iris Samples in Factor Space")
plt.tight_layout()
plt.show()

# Interpretation
This heatmap shows how strongly each original feature contributes to the two latent factors:

Feature	            Factor 1	Factor 2
Sepal length (cm)	0.88	    -0.45
Sepal width (cm)	-0.42	    -0.55
Petal length (cm)	1.00	    0.02
Petal width (cm)	0.96	    0.06

✅ Interpretation:
Factor 1:

Strongly loads on petal length and petal width (close to 1).
Also captures some signal from sepal length (0.88).
Likely represents overall flower size or species-related structure.

Factor 2:

Mild loading and likely represents sepal width variability, but it's weaker overall.
➡Takeaway: The major variation in the Iris dataset is explained by Factor 1, which is driven by petal features.