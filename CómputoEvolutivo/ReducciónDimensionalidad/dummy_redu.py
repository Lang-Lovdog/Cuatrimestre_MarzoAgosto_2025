import polars as pl

df = pl.read_csv("dummy_clean_balanced_dataset_final.csv")

# Get dummies in one line
dummies = [
    "uid",
    "originh",
    "originp",
    "responh",
    "responp",
    "traffic_category",
]

for col in dummies:
    # For all columns in dummies, transform to dummies
    df = df.to_dummies(col)


print(df) 

# Save
df.write_csv("dummy_clean_balanced_dataset_final.csv")
