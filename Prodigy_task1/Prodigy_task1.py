import pandas as pd
import matplotlib.pyplot as plt
import pycountry

# Read the population dataset
df = pd.read_csv("population.csv", skiprows=4)

# Get valid country codes
valid_codes = {country.alpha_3 for country in pycountry.countries}

# Keep only actual countries
country_data = df[df["Country Code"].isin(valid_codes)].copy()

# Convert 2024 population values to numbers
country_data["2024"] = pd.to_numeric(
    country_data["2024"],
    errors="coerce"
)

# Remove missing population values
country_data = country_data.dropna(subset=["2024"])

# Create histogram
plt.figure(figsize=(10, 6))

plt.hist(
    country_data["2024"],
    bins=20,
    edgecolor="black"
)

# Add title and labels
plt.title("Distribution of Population Across Countries (2024)")
plt.xlabel("Population")
plt.ylabel("Number of Countries")

# Add grid
plt.grid(axis="y", alpha=0.25)

# Adjust layout
plt.tight_layout()

# Save the graph
plt.savefig(
    "population_distribution_2024.png",
    dpi=300
)

# Display the graph
plt.show()
