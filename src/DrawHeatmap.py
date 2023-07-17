import folium
import folium.plugins as plugins
import pandas as pd

def drawheatmap():
    df = pd.read_csv('data/cleaned.csv')
    belgium_map = folium.Map(location=[50.5039, 4.4699], zoom_start=8)

    # Create a list of latitudes, longitudes, and prices from the dataset
    locations = df[['latitude', 'longitude', 'price']].values.tolist()

    # Normalize the prices for better heatmap visualization
    max_price = df['price'].max()
    normalized_prices = df['price'] / max_price

    # Add a normalized heatmap layer to the map based on the house prices
    heatmap_layer = plugins.HeatMap(
        locations, min_opacity=0.3, blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
        scale_radius=False, radius=15, weights=normalized_prices.tolist()  # Convert Series to list
    )
    heatmap_layer.add_to(belgium_map)

    # Save the map as an HTML file
    belgium_map.save('belgian_heatmap_house_prices.html')


def drawfullmap():
    df = pd.read_csv('data/cleaned.csv')
    belgium_map = folium.Map(location=[50.5039, 4.4699], zoom_start=8)

    # Iterate over each row in the dataset and add markers to the map
    for index, row in df.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        price = row['price']

        # Add a marker to the map for each location
        folium.Marker(
            location=[latitude, longitude],
            popup=f"Price: {price}",
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(belgium_map)

    # Normalize the prices for better heatmap visualization
    max_price = df['price'].max()
    normalized_prices = df['price'] / max_price

    # Add a normalized heatmap layer to the map based on the house prices
    heatmap_layer = plugins.HeatMap(
        df[['latitude', 'longitude']].values.tolist(),
        min_opacity=0.3, blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
        scale_radius=False, radius=15, weights=normalized_prices.tolist()  # Convert Series to list
    )
    heatmap_layer.add_to(belgium_map)

    belgium_map.save('belgian_fullmap.html')


def drawheatmap_priceperm2(df):
    belgium_map = folium.Map(location=[50.5039, 4.4699], zoom_start=8)

    # Create a list of latitudes, longitudes, and prices from the dataset
    locations = df[['latitude', 'longitude', 'price_per_area_m2']].values.tolist()

    # Add a normalized heatmap layer to the map based on the house prices
    heatmap_layer = plugins.HeatMap(
        locations, min_opacity=0.3, blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'},
        scale_radius=False, radius=15,  # Convert Series to list
    )
    heatmap_layer.add_to(belgium_map)

    # Save the map as an HTML file
    belgium_map.save('belgian_heatmap_priceperm2.html')