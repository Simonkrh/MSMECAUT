import pandas as pd
import matplotlib.pyplot as plt
import calendar


def plot_weekly_rainfall(file_path="Rainfall_vigra.csv"):
    df = pd.read_csv(file_path, sep=";", decimal=",")

    df["Tid(norsk normaltid)"] = pd.to_datetime(
        df["Tid(norsk normaltid)"], format="%d.%m.%Y"
    )

    df.rename(columns={"Nedbør (døgn)": "rainfall"}, inplace=True)

    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    weekly_rainfall = (
        df.set_index("Tid(norsk normaltid)")["rainfall"].resample("W").sum()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        weekly_rainfall.index,
        weekly_rainfall.values,
        label="Weekly Rainfall",
    )
    plt.title("Weekly Rainfall at Vigra Weather Station (2013–2023)")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.show()


def plot_hurricane_events(file_path="Wind_vigra.csv"):
    df = pd.read_csv(file_path, sep=";", decimal=",", dtype=str)

    df["Tid(norsk normaltid)"] = pd.to_datetime(
        df["Tid(norsk normaltid)"], format="%d.%m.%Y"
    )

    col = "Høyeste vindkast (døgn)"

    clean = (
        df[col]
        .astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)")[0]
    )

    df[col] = pd.to_numeric(clean, errors="coerce")

    hurricane_days = df[df["Høyeste vindkast (døgn)"] > 32.6].copy()

    if hurricane_days.empty:
        print("No hurricane-level events found in dataset.")
        return

    hurricane_days.sort_values("Tid(norsk normaltid)", inplace=True)

    hurricane_days["event_id"] = (
        hurricane_days["Tid(norsk normaltid)"].diff().dt.days > 1
    ).cumsum()

    events = hurricane_days.groupby("event_id").first()
    events["year"] = events["Tid(norsk normaltid)"].dt.year

    yearly_counts = events["year"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(yearly_counts.index.astype(str), yearly_counts.values)
    plt.title("Hurricane-Level Wind Events (>32.6 m/s) per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")

    plt.show()


def plot_average_temp_by_month(
    file_path="Temperature_vigra.csv",
    month=1,
    year_start=2013,
    year_end=2023,
    show_errorbars=False,
):
    if not (1 <= int(month) <= 12):
        raise ValueError("Invalid month. Please enter a number between 1 and 12.")

    date_col = "Tid(norsk normaltid)"
    mean_col = "Middeltemperatur (døgn)"

    na_tokens = ["-", "–", "−", "", " ", "NaN"]
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        na_values=na_tokens,
        keep_default_na=True,
    )

    df[date_col] = pd.to_datetime(df[date_col], format="%d.%m.%Y", errors="coerce")

    s = df[mean_col].astype(str)
    s = s.str.replace("\u2212", "-", regex=False)
    s = s.str.replace(",", ".", regex=False).str.strip()
    s = s.mask(s.isin(["-", "–", "−", ""]))
    df["mean_temp"] = pd.to_numeric(s, errors="coerce")

    df = df.dropna(subset=[date_col, "mean_temp"])

    df = df[
        (df[date_col].dt.month == month)
        & (df[date_col].dt.year.between(year_start, year_end))
    ]
    if df.empty:
        print("No data found for the chosen month/year range.")
        return

    yearly = (
        df.groupby(df[date_col].dt.year)["mean_temp"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )

    yearly["std"] = yearly["std"].fillna(0.0)

    x = yearly.index.values
    y = yearly["mean"].values
    s = yearly["std"].values

    plt.figure(figsize=(12, 5))
    if show_errorbars:
        plt.errorbar(x, y, yerr=s, fmt="o-", capsize=5, label="Average ±1 std dev")
    else:
        plt.plot(x, y, marker="o", label="Average temperature")
        plt.fill_between(x, y - s, y + s, alpha=0.2, label="±1 standard deviation")

    plt.title(
        f"Average Daily Mean Temperature in {calendar.month_name[month]} "
        f"({year_start}–{year_end})"
    )
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.show()


if __name__ == "__main__":
    plot_average_temp_by_month(month=1)
