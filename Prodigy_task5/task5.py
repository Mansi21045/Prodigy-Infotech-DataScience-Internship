import pandas as pd
import matplotlib.pyplot as plt
import folium

df = pd.read_csv("task5_accidents.csv")

df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(df["time"], format="%H:%M")
df["hour"] = df["time"].dt.hour

def get_time_period(hour):
    if hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 18:
        return "Afternoon"
    else:
        return "Evening"

df["time_period"] = df["hour"].apply(get_time_period)

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

analyses = [
    ("time_period", "Accidents by Time of Day"),
    ("weather", "Accidents by Weather"),
    ("road_condition", "Accidents by Road Condition"),
    ("severity", "Accident Severity"),
    ("borough", "Accidents by Borough"),
    ("contributing_factor", "Contributing Factors")
]

for column, title in analyses:
    counts = df[column].value_counts()
    plt.figure(figsize=(8,5))
    plt.bar(counts.index, counts.values, edgecolor="black")
    plt.title(title)
    plt.xlabel(column.replace("_"," ").title())
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Interactive accident hotspot map
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        popup=f"{row['borough']} - {row['severity']}",
        fill=True
    ).add_to(nyc_map)

nyc_map.save("accident_hotspots.html")
df.to_csv("task5_cleaned_accidents.csv", index=False)

print("\nTask 5 completed successfully!")
