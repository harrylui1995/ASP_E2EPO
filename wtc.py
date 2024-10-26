import pandas as pd
df = pd.read_csv('filtered_flight_data_all_50.csv')
# Load the aircraft database
aircraft_db = pd.read_csv('data/aircraftDatabase.csv', on_bad_lines='warn', engine='python')
aircraft_db.columns = aircraft_db.columns.str.strip("'")

# Merge the dataframes on icao24
df = df.merge(aircraft_db[['icao24', 'typecode']], on='icao24', how='left')

# print(df['typecode'])

def identify_wtc(typecode):
    # Load the doc8643AircraftTypes.csv file
    aircraft_types_db = pd.read_csv('data/doc8643AircraftTypes.csv')
    
    # Create a dictionary mapping Designator to WTC
    wtc_dict = dict(zip(aircraft_types_db['Designator'], aircraft_types_db['WTC']))
    
    if pd.isna(typecode):
        return 'M'  # Assign 'M' instead of 'Unknown' for NaN values
    
    typecode = str(typecode).upper()
    
    # Check if the typecode exists in our dictionary
    if typecode in wtc_dict:
        return wtc_dict[typecode]
    else:
        # Manual mapping for specific aircraft types
        jumbo = ['A388']
        heavy = ['A359', 'B788', 'B789', 'B77W', 'B772', 'B763', 'A332', 'A333', 'A339', 'B77L', 'A343', 'A310', 'B752', 'B773', 'B78X']
        medium = ['A320', 'B738', 'A319', 'A21N', 'A20N', 'B38M', 'A321', 'BCS3', 'E190', 'B739', 'E295', 'DH8D', 'E195', 'AT76', 'B39M', 'AT75', 'B737', 'CRJ2', 'E290', 'E145', 'SB91']
        light = ['B06', 'C42', 'DA42', 'C208', 'CL60', 'PC12', 'E35L', 'C68A', 'PA46', 'SPIT', 'CRUZ', 'C152', 'P68', 'GLF6', 'EC35', 'C25M', 'PA34', 'C172']
        
        if typecode in jumbo:
            return 'J'
        elif typecode in heavy:
            return 'H'
        elif typecode in medium:
            return 'M'
        elif typecode in light:
            return 'L'
        else:
            return 'M'  # Assign 'M' if still unknown after manual mapping

# Assign WTC based on typecode
df['wtc'] = df['typecode'].apply(identify_wtc)

# Print WTC distribution
print("\nWake Turbulence Category Distribution:")
print(df['wtc'].value_counts(normalize=True) * 100)

# Print the number of unknown WTC
unknown_count = df['wtc'].value_counts().get('Unknown', 0)
print(f"\nNumber of aircraft with unknown WTC: {unknown_count}")

# Optional: If there are many unknowns, you might want to investigate
if unknown_count > 0:
    print("\nSample of aircraft with unknown WTC:")
    print(df[df['wtc'] == 'Unknown'][['icao24', 'typecode']].sample(min(5, unknown_count)))

df.to_csv('filtered_flight_data_all_50.csv')