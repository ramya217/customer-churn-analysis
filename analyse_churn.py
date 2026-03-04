import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Set plot style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Churn Analysis", fontsize=16, fontweight='bold')

# Plot 1 - Churn by Contract Type
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
contract_churn.plot(kind='bar', ax=axes[0,0], color=['#2e6da4', '#f0a500'])
axes[0,0].set_title("Churn by Contract Type")
axes[0,0].set_xlabel("Contract Type")
axes[0,0].set_ylabel("Number of Customers")
axes[0,0].tick_params(axis='x', rotation=0)

# Plot 2 - Churn by Tenure
df.boxplot(column='tenure', by='Churn', ax=axes[0,1], 
           boxprops=dict(color='#2e6da4'))
axes[0,1].set_title("Tenure vs Churn")
axes[0,1].set_xlabel("Churn")
axes[0,1].set_ylabel("Months with Company")

# Plot 3 - Churn by Monthly Charges
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1,0],
           boxprops=dict(color='#f0a500'))
axes[1,0].set_title("Monthly Charges vs Churn")
axes[1,0].set_xlabel("Churn")
axes[1,0].set_ylabel("Monthly Charges ($)")

# Plot 4 - Churn by Internet Service
internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
internet_churn.plot(kind='bar', ax=axes[1,1], color=['#2e6da4', '#f0a500'])
axes[1,1].set_title("Churn by Internet Service")
axes[1,1].set_xlabel("Internet Service Type")
axes[1,1].set_ylabel("Number of Customers")
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved as churn_analysis.png")