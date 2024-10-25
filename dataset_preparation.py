import pandas as pd
import glob
import os
from pyproj import Geod
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def haversine_distance(lat1, lon1, lat2, lon2):
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance / 1852  # Convert meters to nautical miles

def process_parquet_files(directory, airport_lat, airport_lon):
    all_files = glob.glob(os.path.join(directory, "*.parquet"))
    logging.info(f"Found {len(all_files)} parquet files in {directory}")
    
    dataset = []
    
    for file in tqdm(all_files, desc="Processing files"):
        logging.info(f"Processing file: {file}")
        df = pd.read_parquet(file)
        
        flight_count = 0
        skipped_count = 0
        
        for callsign, group in df.groupby('callsign'):
            group = group.sort_values('time')
            
            # Calculate distance to airport
            group['distance_to_airport'] = group.apply(
                lambda row: haversine_distance(row['lat'], row['lon'], airport_lat, airport_lon),
                axis=1
            )
            
            # Find the index where the flight enters 80NM
            entry_index = group[(group['distance_to_airport'] <= 80)&(group['distance_to_airport'] >= 78)].index.min()
            
            if pd.isna(entry_index):
                skipped_count += 1
                continue # Skip if the flight never enters 50NM
            
            # Get the row at 50NM entry
            entry_row = group.loc[entry_index]
            
            # Find the arrival time (the time of the first occurrence of the lowest baroaltitude in the flight)
            lowest_baroaltitude = group['baroaltitude'].min()
            arrival_time = group[group['baroaltitude'] == lowest_baroaltitude]['time'].iloc[0]
            
            # Calculate transit time
            transit_time = arrival_time - entry_row['time']
            
            # Extract required features
            features = entry_row[['lat', 'lon', 'velocity', 'heading', 'vertrate', 
                                  'baroaltitude', 'geoaltitude', 'hour']].to_dict()
            
            # Add to dataset
            dataset.append({
                'callsign': callsign,
                'time': entry_row['time'],
                'icao24': entry_row['icao24'],
                **features,
                'arrival_time': arrival_time,
                'transit_time': transit_time
            })
            
            flight_count += 1
        
        logging.info(f"Processed {flight_count} flights, skipped {skipped_count} flights in {file}")
    
    logging.info(f"Total flights processed: {len(dataset)}")
    return pd.DataFrame(dataset)


# Usage
if __name__ == '__main__':

    airport_lat, airport_lon = 51.1537, -0.1821  # EGKK (London Gatwick) coordinates
    parquet_directory = 'EGKK'

    logging.info("Starting parquet file processing")
    result_df = process_parquet_files(parquet_directory, airport_lat, airport_lon)

    # Save the result
    output_file = 'data/flight_dataset_aug_sep.csv'
    logging.info(f"Saving results to {output_file}")
    result_df.to_csv(output_file, index=False)
    logging.info("Processing completed")