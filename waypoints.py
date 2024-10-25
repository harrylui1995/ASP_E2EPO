import csv
from io import StringIO

# The content of the file as a string
file_content = """	N	E	W
TRIPO	51.713056	1.082778	
KOPUL	51.542222	1.137222	
MIDHURST	51.053889		-0.625000
GOODWOOD	50.855278		-0.756667
MAYFIELD	51.017222	0.116111	
SEAFORD	50.760556	0.121944	
HASTY	50.728333	0.533333	
BEXIL	50.708889	0.736944	
KUNAV	50.515000	1.065556	
LARCK	50.911667	0.446667	
KIDLI	51.771389		-1.361389
KENET	51.520556		-1.455000
BEDEK	51.370833		-1.558611
NIGIT	51.313056		-1.170833
BILNI	50.675278		-2.125833
DOMUT	50.261944		-1.669167
KUMIL	50.575278		-1.610833
SOUTHAMPTON	50.955278		-1.345000
BEWLI	50.758611		-1.810278
WILLO	50.985000		-0.191667
HOLLY	50.886667		-0.095000
BEGTO	50.762500		-1.235556
KATHY	50.520556		-1.333333
AVANT	50.820000		-0.938333
ASTRA	50.865556		-0.146389
TIMBA	50.945556	0.261667	
TANET	51.449444	0.925556	
SPEAR	51.576111	0.700278	
DETLING	51.303889	0.597222	
LYDD	50.999722	0.878611	"""

# Create a CSV reader object
csv_reader = csv.reader(StringIO(file_content), delimiter='\t')

# Skip the header row
next(csv_reader)

# Process each row
waypoints = []
for row in csv_reader:
    if len(row) >= 4:
        name = row[0]
        lat = float(row[1]) if row[1] else None
        lon = float(row[2]) if row[2] else float(row[3]) if row[3] else None
        
        if lat is not None and lon is not None:
            waypoints.append({
                'name': name,
                'latitude': lat,
                'longitude': lon
            })

# Print the extracted waypoints
for waypoint in waypoints:
    print(f"Waypoint: {waypoint['name']}")
    print(f"Latitude: {waypoint['latitude']}")
    print(f"Longitude: {waypoint['longitude']}")
    print()

    

    # Define the output CSV file name
output_file = 'waypoints.csv'

# Open the file in write mode
with open(output_file, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['Name', 'Latitude', 'Longitude'])
    
    # Write each waypoint to the CSV file
    for waypoint in waypoints:
        csv_writer.writerow([
            waypoint['name'],
            waypoint['latitude'],
            waypoint['longitude']
        ])

print(f"Waypoints have been saved to {output_file}")
