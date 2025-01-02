import os
import geopandas as gpd

def dissolve_internal_boundaries():
    """
    Dissolves internal boundaries of all GeoJSON files in the current folder,
    overwriting the original files with the dissolved boundaries.
    """
    # Get the current folder path
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # Iterate through all files in the current folder
    for file_name in os.listdir(current_folder):
        if file_name.endswith(".geojson"):
            file_path = os.path.join(current_folder, file_name)

            try:
                # Load GeoJSON file
                gdf = gpd.read_file(file_path)

                # Dissolve internal boundaries (all geometries merged into one boundary)
                dissolved = gdf.dissolve(by=None)

                # Overwrite the original file with the dissolved geometry
                dissolved.to_file(file_path, driver="GeoJSON")

                print(f"Successfully processed and overwritten: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Call the function
if __name__ == "__main__":
    dissolve_internal_boundaries()
