import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(32)

# Create a sample dataset: 10 tours (clusters), each with 10 participants
data = {
    'Tour_ID': np.repeat(np.arange(1, 11), 10),  # 10 tours, each with 10 participants
    'Participant_ID': np.arange(1, 101),  # 100 participants total
    'Satisfaction_Score': np.random.randint(1, 10, size=100)  # Scores between 1 and 9
}

df = pd.DataFrame(data)

# Step 1: Display the initial dataset
print("Tour Experience Dataset (First 20 Rows):")
print(df.head(20))

# Step 2: Randomly select a few clusters (tours)
selected_clusters = np.random.choice(df['Tour_ID'].unique(), size=3, replace=False)
print("\nSelected Clusters (Tours):", selected_clusters)

# Step 3: Filter the DataFrame to include only the selected clusters
cluster_sample = df[df['Tour_ID'].isin(selected_clusters)]

# Step 4: Calculate the average satisfaction score from the sampled clusters
average_satisfaction = cluster_sample['Satisfaction_Score'].mean()
print(f"\nAverage Satisfaction Score (Sampled Clusters): {average_satisfaction:.2f}")

# Step 5: Plot the results using Seaborn and Matplotlib

# Set up a figure with two plots: One for the sampled clusters, and one for all clusters
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Plot satisfaction scores for all clusters
sns.boxplot(data=df, x='Tour_ID', y='Satisfaction_Score', ax=axes[0], palette="Blues")
axes[0].set_title("Satisfaction Scores for All Tours")
axes[0].set_xlabel("Tour ID")
axes[0].set_ylabel("Satisfaction Score")

# Plot satisfaction scores for the selected clusters
sns.boxplot(data=cluster_sample, x='Tour_ID', y='Satisfaction_Score', ax=axes[1], palette="Greens")
axes[1].set_title("Satisfaction Scores for Sampled Tours")
axes[1].set_xlabel("Tour ID")

# Add a horizontal line showing the average satisfaction score from the sampled clusters
axes[1].axhline(average_satisfaction, color='red', linestyle='--', label='Average Score')
axes[1].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
