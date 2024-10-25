import pandas as pd

df = pd.read_csv('flight_dataset.csv')
print(df.shape)
# Convert transit_time to seconds
# Remove any leading/trailing whitespace and newline characters
df['transit_time'] = df['transit_time'].str.strip()

# Convert transit_time to seconds, handling potential errors
df['transit_time'] = pd.to_timedelta(df['transit_time'], errors='coerce').dt.total_seconds()

# Replace NaN values with a placeholder or drop them based on your requirements
df['transit_time'] = df['transit_time'].fillna(-1)  # or use df = df.dropna(subset=['transit_time'])
df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce')
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['baroaltitude'] = pd.to_numeric(df['baroaltitude'], errors='coerce')
df['geoaltitude'] = pd.to_numeric(df['geoaltitude'], errors='coerce')
df['vertrate'] = pd.to_numeric(df['vertrate'], errors='coerce')
print(df['transit_time'].dtype)
print(df['transit_time'].head())


df = df[(df['velocity']<=250)&(df['velocity']>=100)]
df = df[(df['baroaltitude']<=6000)&(df['baroaltitude']>=2000)]
df = df[(df['vertrate']<=0)&(df['vertrate']>=-15)]
df = df[(df['transit_time']<=2500)&(df['transit_time']>500)] 
df = df.dropna().reset_index(drop=True)  # Drop NaN values and reset index
print(len(df))
print(df.columns)