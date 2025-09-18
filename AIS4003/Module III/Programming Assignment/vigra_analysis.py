import pandas as pd
import matplotlib.pyplot as plt
import calendar

DATE_COLUNM = "Tid(norsk normaltid)"


def plot_weekly_rainfall(file_path="Rainfall_vigra.csv"):
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=[DATE_COLUNM],
        dayfirst=True,
    )

    weekly_rainfall = df.set_index(DATE_COLUNM)["Nedbør (døgn)"].resample("W").sum()

    plt.figure(figsize=(12, 5))
    plt.plot(weekly_rainfall.index, weekly_rainfall.values, label="Weekly Rainfall")
    plt.title("Weekly Rainfall at Vigra Weather Station (2013–2023)")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.show()


def plot_hurricane_events(file_path="Wind_vigra.csv"):
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=[DATE_COLUNM],
        dayfirst=True,
        na_values=["-", "NaN"],
    )
    column = "Høyeste vindkast (døgn)"

    hurricane_days = df[df[column] > 32.6].copy()
    if hurricane_days.empty:
        print("No hurricane-level events found in dataset.")
        return

    hurricane_days.sort_values(DATE_COLUNM, inplace=True)
    hurricane_days["event_id"] = (
        hurricane_days[DATE_COLUNM].diff().dt.days > 1
    ).cumsum()

    events = hurricane_days.groupby("event_id").first()
    yearly_counts = events[DATE_COLUNM].dt.year.value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(yearly_counts.index.astype(str), yearly_counts.values)
    plt.title("Hurricane-Level Wind Events (>32.6 m/s) per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.show()


def plot_average_temp_by_month(
    file_path="Temperature_vigra.csv", month=1, year_start=2013, year_end=2023
):
    month = int(month)
    if not (1 <= month <= 12):
        raise ValueError("Invalid month. Please enter a number between 1 and 12.")

    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=[DATE_COLUNM],
        dayfirst=True,
        na_values=["-", "NaN"],
    )
    mean_column = "Middeltemperatur (døgn)"

    selection = (df[DATE_COLUNM].dt.month == month) & df[DATE_COLUNM].dt.year.between(
        year_start, year_end
    )
    df = df.loc[selection, [DATE_COLUNM, mean_column]].dropna()
    if df.empty:
        print("No data found for the chosen month/year range.")
        return

    yearly = (
        df.groupby(df[DATE_COLUNM].dt.year)[mean_column]
        .agg(["mean", "std", "count"])
        .sort_index()
        .fillna({"std": 0.0})
    )

    x = yearly.index.values
    y = yearly["mean"].values
    s = yearly["std"].values

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", label="Average temperature")
    plt.fill_between(x, y - s, y + s, alpha=0.2, label="±1 standard deviation")
    plt.title(
        f"Average Daily Mean Temperature in {calendar.month_name[month]} ({year_start}–{year_end})"
    )
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.show()


if __name__ == "__main__":
    # plot_weekly_rainfall()
    # plot_hurricane_events()
    plot_average_temp_by_month(year_start=2010, year_end=2015, month=5)
