import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the train data
df = pd.read_csv('amd_train_atr0.csv')

# Create a figure with multiple subplots to verify the data is different
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: First 1000 points of atr_0
ax1 = axes[0, 0]
ax1.plot(df['atr_0'].head(1000).values, label='atr_0', color='blue', alpha=0.7)
ax1.set_title('First 1000 points of atr_0')
ax1.set_xlabel('Index')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# Plot 2: First 1000 points of atr_90
ax2 = axes[0, 1]
ax2.plot(df['atr_90'].head(1000).values, label='atr_90', color='green', alpha=0.7)
ax2.set_title('First 1000 points of atr_90')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)

# Plot 3: Overlay both on the same plot
ax3 = axes[1, 0]
ax3.plot(df['atr_0'].head(1000).values, label='atr_0', color='blue', alpha=0.5, linewidth=2)
ax3.plot(df['atr_90'].head(1000).values, label='atr_90', color='green', alpha=0.5, linewidth=2)
ax3.set_title('Overlay: atr_0 vs atr_90 (first 1000 points)')
ax3.set_xlabel('Index')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Scatter plot of atr_0 vs atr_90
ax4 = axes[1, 1]
sample_indices = np.random.choice(len(df), 5000, replace=False)
ax4.scatter(df['atr_0'].iloc[sample_indices], df['atr_90'].iloc[sample_indices],
            alpha=0.3, s=1)
ax4.plot([0, df[['atr_0', 'atr_90']].max().max()],
         [0, df[['atr_0', 'atr_90']].max().max()],
         'r--', label='y=x line')
ax4.set_title('Scatter: atr_0 vs atr_90 (5000 random points)')
ax4.set_xlabel('atr_0')
ax4.set_ylabel('atr_90')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Histogram comparison
ax5 = axes[2, 0]
ax5.hist(df['atr_0'], bins=50, alpha=0.5, label='atr_0', color='blue')
ax5.hist(df['atr_90'], bins=50, alpha=0.5, label='atr_90', color='green')
ax5.set_title('Histogram Comparison')
ax5.set_xlabel('Value')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)

# Plot 6: Difference plot
ax6 = axes[2, 1]
diff = (df['atr_0'] - df['atr_90']).head(1000)
ax6.plot(diff.values, color='red', alpha=0.7)
ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax6.set_title('Difference: atr_0 - atr_90 (first 1000 points)')
ax6.set_xlabel('Index')
ax6.set_ylabel('Difference')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_verification.png', dpi=100)
print("Verification plot saved to 'data_verification.png'")

# Print statistics
print("\n=== Data Verification ===")
print(f"Total rows: {len(df)}")
print(f"\natr_0 statistics:")
print(f"  Mean: {df['atr_0'].mean():.6f}")
print(f"  Std:  {df['atr_0'].std():.6f}")
print(f"  Min:  {df['atr_0'].min():.6f}")
print(f"  Max:  {df['atr_0'].max():.6f}")

print(f"\natr_90 statistics:")
print(f"  Mean: {df['atr_90'].mean():.6f}")
print(f"  Std:  {df['atr_90'].std():.6f}")
print(f"  Min:  {df['atr_90'].min():.6f}")
print(f"  Max:  {df['atr_90'].max():.6f}")

print(f"\nDifference statistics:")
diff_all = df['atr_0'] - df['atr_90']
print(f"  Mean: {diff_all.mean():.6f}")
print(f"  Std:  {diff_all.std():.6f}")
print(f"  Min:  {diff_all.min():.6f}")
print(f"  Max:  {diff_all.max():.6f}")

print(f"\nCorrelation: {df['atr_0'].corr(df['atr_90']):.6f}")
