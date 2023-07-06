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
    belgium_map.save('belgian_price_heatmap.html')


if __name__ == "__main__":
    drawheatmap()