from matplotlib.colors import to_rgba
import pandas as pd
import matplotlib.pyplot as plt

def plot_comparative_temperature(results):
    """
    Plots the maximum & minimum temperatures for each year, for each city.

    Parameters:
        results (dict): Dictionary containing processed data for each city.
    """

    plt.figure(figsize=(12, 8))

    base_colors = {
        "Madrid": "red",
        "London": "blue",
        "Rio": "green"
    }

    for city, data in results.items():
        annual_max_temp = data["temperature_2m_mean"].resample("YE").max()
        annual_min_temp = data["temperature_2m_mean"].resample("YE").min()

        max_color = to_rgba(base_colors[city], alpha=1.0)
        min_color = to_rgba(base_colors[city], alpha=0.5) 

        plt.plot(annual_max_temp.index, annual_max_temp, label=f"{city} (max)", color=max_color)
        plt.plot(annual_min_temp.index, annual_min_temp, label=f"{city} (min)", color=min_color)
    
    plt.title('Comparative Temperatures')
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Legend")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_comparative_precipitation(results):
    """
    Plots the accumulated precipitation for each semester (April-September and October-March) for each city.

    Parameters:
        results (dict): Dictionary containing processed data for each city.
    """
    semesters = {
        "April-September": (4, 10),  
        "October-March": (10, 4)  
    }

    colors = {
        "Madrid": "red",
        "London": "blue",
        "Rio": "green"
    }

    plt.figure(figsize=(14, 8))

    for city, data in results.items():
        semester_precipitation = []

        for year in range(data.index.year.min(), data.index.year.max() + 1):
            for semester, (start_month, end_month) in semesters.items():
                if start_month < end_month:
                    # Semester within same year
                    semester_data = data[
                        (data.index.year == year) &
                        (data.index.month >= start_month) &
                        (data.index.month < end_month)
                    ]
                else:
                    # Semester changing year
                    semester_data = data[
                        ((data.index.year == year) & (data.index.month >= start_month)) |
                        ((data.index.year == year + 1) & (data.index.month < end_month))
                    ]

                accumulated_precipitation = semester_data["precipitation_sum"].sum()
                semester_precipitation.append({
                    "city": city,
                    "semester": semester,
                    "year": year,
                    "precipitation": accumulated_precipitation
                })

        semester_df = pd.DataFrame(semester_precipitation)

        for semester in semesters.keys():
            semester_data = semester_df[semester_df["semester"] == semester]
            offset = 0.25 if semester == "April-September" else -0.25
            plt.bar(
                semester_data["year"] + offset,
                semester_data["precipitation"],
                width=0.4,
                alpha=0.5,
                color=colors[city], 
                label=f"{city} ({semester})" if semester == "April-September" else None
            )

    plt.title("Accumulated Precipitation by Semester Over Time")
    plt.xlabel("Year")
    plt.ylabel("Accumulated Precipitation (mm)")
    plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
