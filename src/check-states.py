import pandas as pd

# Load the cleaned data
df = pd.read_csv('../data/processed/cherry_blossom_2025_cleaned.csv')

# Get all unique US "states"
us_runners = df[df['Is_US'] == True]
unique_states = us_runners['State'].value_counts()

print("All US 'States' found:")
print(unique_states)
print(f"\nTotal unique: {len(unique_states)}")

# Check if any look suspicious
print("\n'States' with only 1-5 runners (might be errors):")
print(unique_states[unique_states <= 5])