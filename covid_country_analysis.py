
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "country_wise_latest.csv"  
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", list(df.columns))

def pick_first_matching(colnames, candidates):
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

country_col = pick_first_matching(df.columns, ["Country/Region", "Country", "Location", "location", "Country_Name"])
confirmed_col = pick_first_matching(df.columns, ["Confirmed", "TotalConfirmed", "Total Cases", "Total_Cases", "Cases"])
deaths_col = pick_first_matching(df.columns, ["Deaths", "TotalDeaths", "Total Deaths", "Total_Deaths"])
recovered_col = pick_first_matching(df.columns, ["Recovered", "TotalRecovered", "Total Recovered", "Total_Recovered"])
active_col = pick_first_matching(df.columns, ["Active", "Active Cases", "Active_Cases"])
region_col = pick_first_matching(df.columns, ["WHO Region", "Region"])

required_any = [country_col, confirmed_col, deaths_col]
if any(x is None for x in required_any):
    raise ValueError(f"Missing required columns. Detected -> country: {country_col}, confirmed: {confirmed_col}, deaths: {deaths_col}")

df[country_col] = df[country_col].astype(str).str.strip()

for c in [confirmed_col, deaths_col, recovered_col, active_col]:
    if c is not None and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in [confirmed_col, deaths_col, recovered_col, active_col]:
    if c is not None and c in df.columns:
        df[c] = df[c].fillna(0)

if recovered_col is not None and recovered_col in df.columns:
    df["Recovery Rate (%)"] = (df[recovered_col] / df[confirmed_col]).where(df[confirmed_col] > 0, np.nan) * 100

df["Death Rate (%)"] = (df[deaths_col] / df[confirmed_col]).where(df[confirmed_col] > 0, np.nan) * 100

print("\nHead:")
try:
    from IPython.display import display
    display(df.head())
except Exception:
    print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nBasic stats for numeric columns:")
print(df.describe())

# 1) Top 10 by Confirmed
top_confirmed = df.sort_values(by=confirmed_col, ascending=False).head(10)
plt.figure()
plt.bar(top_confirmed[country_col], top_confirmed[confirmed_col])
plt.title("Top 10 Countries by Confirmed Cases")
plt.xlabel("Country")
plt.ylabel("Confirmed Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 2) Top 10 by Deaths
top_deaths = df.sort_values(by=deaths_col, ascending=False).head(10)
plt.figure()
plt.bar(top_deaths[country_col], top_deaths[deaths_col])
plt.title("Top 10 Countries by Deaths")
plt.xlabel("Country")
plt.ylabel("Deaths")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 3) Histogram of Confirmed
plt.figure()
plt.hist(df[confirmed_col].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4) Scatter: Deaths vs Confirmed
plt.figure()
plt.scatter(df[confirmed_col], df[deaths_col], alpha=0.7)
plt.title("Deaths vs Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.tight_layout()
plt.show()

# Optional by Region
if region_col is not None and region_col in df.columns:
    region_totals = df.groupby(region_col, as_index=False)[[confirmed_col, deaths_col]].sum(numeric_only=True)
    plt.figure()
    plt.bar(region_totals[region_col], region_totals[confirmed_col])
    plt.title("Total Confirmed by Region")
    plt.xlabel("Region")
    plt.ylabel("Confirmed Cases")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

top10_cases = df.sort_values(by=confirmed_col, ascending=False)[[country_col, confirmed_col]].head(10)
top10_deaths = df.sort_values(by=deaths_col, ascending=False)[[country_col, deaths_col]].head(10)

print("\nTop 10 by Confirmed Cases:\n", top10_cases.to_string(index=False))
print("\nTop 10 by Deaths:\n", top10_deaths.to_string(index=False))

lines = []
lines.append("COVID-19 Country-Level Snapshot Summary\n")
lines.append("Top 10 Countries by Confirmed Cases:\n")
lines.extend(top10_cases.to_string(index=False).splitlines())
lines.append("\nTop 10 Countries by Deaths:\n")
lines.extend(top10_deaths.to_string(index=False).splitlines())

if "Recovery Rate (%)" in df.columns:
    best_recovery = df.dropna(subset=["Recovery Rate (%)"]).sort_values("Recovery Rate (%)", ascending=False).head(5)
    lines.append("\nBest Recovery Rates (%):\n")
    lines.extend(best_recovery[[country_col, "Recovery Rate (%)"]].to_string(index=False).splitlines())

with open("insights_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nSaved summary to insights_summary.txt")
