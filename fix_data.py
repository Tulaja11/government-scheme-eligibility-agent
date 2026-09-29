import pandas as pd

df = pd.read_csv("updated_data.csv")

# Drop useless columns
df = df.drop(columns=["Unnamed: 9"])

# Remove schemes with very short eligibility
df = df[df["eligibility"].str.len() >= 50]

# Create source URL from slug
df["source_url"] = "https://www.myscheme.gov.in/schemes/" + df["slug"].fillna("")

# Drop slug (we have the URL now)
df = df.drop(columns=["slug"])

# Reset index
df = df.reset_index(drop=True)

# Save
df.to_csv("data/schemes_clean.csv", index=False)
print(f"Total schemes: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Sample URL: {df.iloc[0]['source_url']}")