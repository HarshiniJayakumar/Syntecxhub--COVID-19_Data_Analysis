import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
data = pd.read_csv("covid-data.csv")
covid = data[["location","date","total_cases","new_cases","total_deaths","population"]].copy()
covid["date"] = pd.to_datetime(covid["date"])
countries = ["India","United States","Brazil","United Kingdom"]
covid = covid[covid["location"].isin(countries)]
for c in ["new_cases","total_cases","total_deaths"]:
    covid[c] = covid[c].fillna(0)

daily_cases = covid.groupby(["location","date"],as_index=False)["new_cases"].sum()

weekly_cases = (daily_cases.set_index("date").groupby("location")["new_cases"].resample("W").sum().reset_index())

daily_cases["rolling_avg"] = (daily_cases.groupby("location")["new_cases"]
                              .transform(lambda x: x.rolling(window=7, min_periods=1).mean()))

daily_cases["R_estimate"] = (daily_cases.groupby("location")["rolling_avg"]
                            .transform(lambda x: (x / x.shift(4)).replace([float("inf"), -float("inf")], pd.NA)))

daily_cases["R_estimate"] = (daily_cases["R_estimate"].fillna(1).clip(lower=0, upper=5).round(2))

summary = covid.groupby("location").agg(
    Total_Cases=("total_cases","max"),
    Total_Deaths=("total_deaths","max"),
    Population=("population","max")).reset_index()

avg_daily = daily_cases.groupby("location")["new_cases"].mean().reset_index(name="Average_Daily_Cases")
peak_daily = daily_cases.groupby("location")["new_cases"].max().reset_index(name="Peak_Daily_Cases")

peak_dates = (daily_cases.groupby("location",group_keys=False)
            .apply(lambda x:x.loc[x["new_cases"].idxmax(),["location","date"]]).reset_index(drop=True)
            .rename(columns={"date":"Peak_Date"}))
avg_r = daily_cases.groupby("location")["R_estimate"].mean().reset_index(name="Average_R")

summary = (summary.merge(avg_daily,on="location")
                  .merge(peak_daily,on="location")
                  .merge(peak_dates,on="location")
                  .merge(avg_r,on="location"))

summary["Average_Daily_Cases"]=summary["Average_Daily_Cases"].round(2)
summary["Average_R"]=summary["Average_R"].round(2)
print("COVID 19 SUMMARY")
print("------------------")
print(summary)
print("\nBasic Reproduction Estimates (Average R):")
print("---------------------------------------------")
for _, row in summary.iterrows():
    r = row["Average_R"]

    if r > 1:
        status = "Increasing spread"
    elif r < 1:
        status = "Declining spread"
    else:
        status = "Stable"

    print(f"{row['location']}: Average R = {r:.2f} ({status})")

with open("covid_analysis_summary.txt","w",encoding="utf-8") as f:
    f.write("COVID-19 ANALYSIS SUMMARY\n")
    f.write("="*120+"\n")
    f.write(summary.to_string(index=False))
    f.write("\n\nInterpretation\n")
    f.write("-"*20+"\n")
    for _,r in summary.iterrows():
        f.write(f"\n{r['location']}\n")
        f.write(f"Peak Date: {r['Peak_Date'].date()}\n")
        f.write(f"Peak Daily Cases: {int(r['Peak_Daily_Cases'])}\n")
        f.write(f"Average Estimated R: {r['Average_R']:.2f}\n")
        if r["Average_R"] > 1:
            status = "Increasing spread"
        elif r["Average_R"] < 1:
            status = "Declining spread"
        else:
            status = "Stable"

        f.write(f"Spread Insight: {status}\n")
with open("conclusions.txt","w") as f:
    f.write("- Daily and weekly cases computed.\n")
    f.write("- Rolling averages smoothed fluctuations.\n")
    f.write("- Peak infection dates identified.\n")
    f.write("- Country comparison completed.\n")
    f.write("- Basic reproduction estimate calculated.\n")

# Daily cases comparison
plt.figure(figsize=(12,6))
for c in countries:
    d = daily_cases[daily_cases.location == c]
    plt.plot(d.date, d["new_cases"], label=c)
plt.title("Daily COVID-19 Cases Comparison")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_cases_comparison.png", dpi=300)
plt.close()

# Rolling average comparison
plt.figure(figsize=(12,6))
for c in countries:
    d = daily_cases[daily_cases.location == c]
    plt.plot(d.date, d["rolling_avg"], label=c)
plt.title("7-Day Rolling Average")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rolling_average.png", dpi=300)
plt.close()

# Weekly cases comparison
plt.figure(figsize=(12,6))
for c in countries:
    d = weekly_cases[weekly_cases.location == c]
    plt.plot(d.date, d.new_cases, label=c)
plt.title("Weekly COVID-19 Cases Comparison")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weekly_cases_comparison.png", dpi=300)
plt.close()

# Peak daily cases bar chart
plt.figure(figsize=(8,5))
plt.bar(summary.location, summary.Peak_Daily_Cases)
plt.title("Peak Daily Cases")
plt.tight_layout()
plt.savefig("peak_daily_cases.png", dpi=300)
plt.close()

# Total cases bar chart
plt.figure(figsize=(8,5))
plt.bar(summary.location, summary.Total_Cases,color="orange")
plt.title("Total Cases")
plt.tight_layout()
plt.savefig("total_cases.png", dpi=300)
plt.close()

