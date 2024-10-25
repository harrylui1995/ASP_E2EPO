import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Load the dataset
df = pd.read_csv('flight_dataset.csv')

# Convert velocity to numeric, replacing any non-numeric values with NaN
df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce')

# Remove rows with NaN velocity
df = df.dropna(subset=['velocity'])

# Set up the map
plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Set the extent of the map (adjust as needed)
ax.set_extent([-1, 2, 50, 52], crs=ccrs.PlateCarree())

# Plot entry points
scatter = ax.scatter(df['lon'], df['lat'], c=df['velocity'], 
                     cmap='viridis', transform=ccrs.PlateCarree(),
                     s=20, alpha=0.6)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('Velocity')

# Plot EGKK airport
ax.plot(-0.1821, 51.1537, 'r*', markersize=15, transform=ccrs.PlateCarree())
ax.text(-0.1821, 51.1537, 'EGKK', fontsize=12, 
        ha='right', va='bottom', transform=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Add title
plt.title('Flight Entry Points (50NM from EGKK)', fontsize=16)

# Save the plot
plt.savefig('entry_points_map.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional visualizations

# 1. Histogram of entry altitudes
plt.figure(figsize=(10, 6))
plt.hist(df['baroaltitude'], bins=30, edgecolor='black')
plt.title('Distribution of Entry Altitudes')
plt.xlabel('Barometric Altitude')
plt.ylabel('Frequency')
plt.savefig('entry_altitude_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter plot of velocity vs. vertical rate
plt.figure(figsize=(10, 6))
plt.scatter(df['velocity'], df['vertrate'], alpha=0.5)
plt.title('Velocity vs. Vertical Rate at Entry Point')
plt.xlabel('Velocity')
plt.ylabel('Vertical Rate')
plt.savefig('velocity_vs_vertrate_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Box plot of transit times by hour
df['hour'] = pd.to_datetime(df['time']).dt.hour
plt.figure(figsize=(12, 6))
df.boxplot(column='transit_time', by='hour')
plt.title('Transit Times by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Transit Time')
plt.savefig('transit_time_by_hour_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations have been saved.")