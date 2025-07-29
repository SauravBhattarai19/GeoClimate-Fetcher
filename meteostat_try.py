from meteostat import Daily
from datetime import datetime

# Set parameters
station_id = "72505"  # Change this to your station ID
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Get tmax data
data = Daily(station_id, start_date, end_date)
df = data.fetch()

# Get only tmax column
tmax_data = df['tmax']

print(tmax_data)
# Save to CSV
tmax_data.to_csv("tmax_data.csv")