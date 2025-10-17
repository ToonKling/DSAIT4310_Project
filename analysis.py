import pandas as pd
import duckdb

airports = pd.read_csv('airports.csv')
airports = airports[airports['Country'] == 'United States']
airports = airports[airports['IATA'].str.len() == 3]
airlines = pd.read_csv('airlines.csv')
routes = pd.read_csv('routes.csv')
nr_rows = 100000 # Total dataset size is 5559465
delays = duckdb.query(f'SELECT * FROM "{'./delays.parquet'}" LIMIT {nr_rows};').df()

# Filter out airports for which we have no flight information
# Given 100000 rows, this leaves us with 60 airports, less than the 349 they have :(
airports = airports[airports['IATA'].isin(delays['ORIGIN'])]

for _, row in airports.iterrows():
    airport_code = row['IATA']
    involving_airport = delays[(delays['ORIGIN'] == airport_code) | (delays['DEST'] == airport_code)].copy()
    delays_involving_airport = involving_airport[involving_airport['CARRIER_DELAY'] > 0]
    mean = delays_involving_airport['CARRIER_DELAY'].mean()
    print(f'We have {len(involving_airport)} data points involving {airport_code}, {len(delays_involving_airport)} are delayed with mean carrier delay: {mean}')

    # Finding vulnerability of this airport
    # Timezones wont be an issue right?
    involving_airport['DEP_SCH'] = (pd.to_datetime(involving_airport['Scheduled_DEP'], errors="coerce").dt.floor("h"))
    involving_airport['DEP_ACT'] = pd.to_datetime(involving_airport['Actual_DEP_dt_EST'], errors="coerce").dt.floor("h")
    involving_airport['ARR_SCH'] = pd.to_datetime(involving_airport['Scheduled_ARR_Ori'], errors="coerce").dt.floor("h")
    involving_airport['ARR_ACT'] = pd.to_datetime(involving_airport['Actual_ARR_dt_Ori'], errors="coerce").dt.floor("h")
    involving_airport = involving_airport[["DEP_SCH", "DEP_ACT", "ARR_SCH", "ARR_ACT"]]
    involving_airport = involving_airport.dropna()

    # TODO: Exclude hours 0-6 for each day, see paper
    events = involving_airport.melt(value_name="timestamp", var_name="event_type")
    counts = events.groupby(["timestamp", "event_type"]).size().reset_index(name="total")
    pivoted = counts.pivot(index="timestamp", columns="event_type", values="total").fillna(0).astype(int)
    pivoted['PLANNED_CAP'] = pivoted['DEP_SCH'] + pivoted['ARR_SCH']
    pivoted['ACTUAL_CAP']  = pivoted['DEP_ACT'] + pivoted['ARR_ACT']
    pivoted['VULN']        = pivoted['ACTUAL_CAP'] > 0.9 * pivoted['PLANNED_CAP'] # Assume alpha = 0.9
    vulnerability = pivoted['VULN'].mean()

    print(f'Vulnerability for {airport_code} is {vulnerability:.2f}\n')

