import os
from flask import Flask, jsonify, render_template, request
import ee
from google.oauth2 import service_account
from datetime import datetime, timedelta
import math  # Import math module for logarithmic calculations
import json
app = Flask(__name__)
from flask_caching import Cache
from functools import lru_cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
# Define the file paths
file_path = r"D:\Download\SSTA-Smart-System-for-Tracking-Airpollution-main\SSTA-Smart-System-for-Tracking-Airpollution-main\flaskapp\config\creds2.json"
default_path = "config/creds2.json"

# Check if the file exists and assign the appropriate value to the variable
SERVICE_ACCOUNT_FILE = file_path if os.path.exists(file_path) else default_path

# Load your Windy API key
WINDY_API_KEY = "DHnqHp6YzeueWA6uhkK3cxT8USF5QsuX"

# Constants for pollutant calculations
g = 9.82           # Acceleration due to gravity (m/s^2)
m_H2O = 0.01801528  # Molar mass of water vapor (kg/mol)
m_dry_air = 0.0289644  # Molar mass of dry air (kg/mol)

# Authenticate to Google Earth Engine using the service account
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
ee.Initialize(credentials)
# Check if credentials file exists before loading
try:
    credentials = service_account.Credentials.from_service_account_file(
        'config/creds2.json'
    )
except Exception as e:
    print("Skipping Google API authentication for local testing:", e)
    credentials = None

def initialize_nltk():
    """Initialize NLTK by downloading required resources with error handling."""
    required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
                # Continue with reduced functionality if download fails
                pass

# Call NLTK initialization at startup
initialize_nltk()
# Function to adjust units based on data range
def adjust_units(min_value, max_value, base_unit):
    prefixes = {
        -12: 'p',
        -9: 'n',
        -6: 'μ',
        -3: 'm',
        0: '',
        3: 'k',
        6: 'M',
        9: 'G',
        12: 'T'
    }

    abs_max = max(abs(min_value), abs(max_value))
    if abs_max == 0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs_max)))
        exponent = (exponent // 3) * 3  # Round to nearest lower multiple of 3
        exponent = min(max(exponent, -12), 12)  # Limit exponent between -12 and 12

    scaling_factor = 10 ** (-exponent)
    prefix = prefixes.get(exponent, '')
    adjusted_unit = f"{prefix}{base_unit}"
    return scaling_factor, adjusted_unit

def interpolate_data_if_empty(pollutant, city_lat, city_lon, start_date, end_date, original_buffer):
    # If no data is available at the given buffer, we try larger buffers
    # This is a simple interpolation approach by expanding the search area until we find data.
    # We do not change any other logic, just attempt to find data in a larger radius and return mean.
    multipliers = [2, 5, 10]  # Try larger and larger radii
    for m in multipliers:
        new_buffer = original_buffer * m
        buffered_city_geometry = ee.Geometry.Point(city_lon, city_lat).buffer(new_buffer)
        if pollutant == 'PM10':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('absorbing_aerosol_index')
        elif pollutant == 'PM2.5':
            filtered_collection = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('Optical_Depth_055')
        elif pollutant == 'NO2':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('NO2_column_number_density')
        elif pollutant == 'CO':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select(['CO_column_number_density', 'H2O_column_number_density'])
        elif pollutant == 'SO2':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('SO2_column_number_density')
        elif pollutant == 'O3':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('O3_column_number_density')
        elif pollutant == 'HCHO':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('tropospheric_HCHO_column_number_density')
        else:
            filtered_collection = None

        if filtered_collection and filtered_collection.size().getInfo() > 0:
            # We found data in a larger area. Take the mean and return it.
            def mask_negative_values(image):
                return image.updateMask(image.gte(0))
            filtered_collection = filtered_collection.map(mask_negative_values)
            mean_image = filtered_collection.mean().clip(buffered_city_geometry)
            # Return this mean image as "interpolated" data
            return mean_image, buffered_city_geometry
    # If no data found in any expanded search, return None
    return None, None

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')

# Route for the about page
@app.route('/about/')
def about():
    return render_template('about.html')

# Route for the sample page
@app.route('/sample/')
def sample():
    return render_template('sample.html')

# Route for the contact page
@app.route('/contact/')
def contact():
    return render_template('contact_us.html')

# API route to fetch pollutant data
@app.route('/api/get-pollutant', methods=['GET'])
def get_pollutant():
    try:
        # Retrieve parameters from the request
        city_lat = float(request.args.get('lat'))
        city_lon = float(request.args.get('lon'))
        buffer = request.args.get('buffer', default=50000, type=int)
        hml = request.args.get('hml', 'false').lower() == 'true'


        # Set default start and end dates (last 7 days)
        current_date = datetime.utcnow()
        default_end_date = current_date.strftime('%Y-%m-%dT%H:%M:%S')
        default_start_date = (current_date - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')

        start_date = request.args.get('start_date', default_start_date)
        end_date = request.args.get('end_date', default_end_date)

        pollutant = request.args.get('pollutant', 'PM2.5')  # Default to PM2.5 if not specified

        if not city_lat or not city_lon:
            return jsonify({'error': 'Latitude and longitude are required parameters.'}), 400

        # Define a buffer around the specified point
        buffer_radius = buffer  # in meters
        buffered_city_geometry = ee.Geometry.Point(city_lon, city_lat).buffer(buffer_radius)

        if pollutant == 'PM10':
            # Fetch the aerosol index data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('absorbing_aerosol_index')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('PM10', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No PM10 data available for the specified parameters.'}), 404
                else:
                    aerosol_index_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                aerosol_index_mean = filtered_collection.mean().clip(buffered_city_geometry)

            # Convert aerosol index to PM10 concentration
            PM10_mean = aerosol_index_mean.multiply(50).add(20)  # Adjust scaling factor and offset as needed
            pollutant_mean = PM10_mean.rename('PM10')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('PM10_min', None)
            max_value = stats.get('PM10_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for PM10.'}), 500

            base_unit = 'µg/m³'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('PM10_p5', None)
            max_value = percentiles.get('PM10_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for PM10.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = adjusted_unit

        elif pollutant == 'PM2.5':
            # Fetch and process the PM2.5 data using the MODIS dataset
            filtered_collection = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('Optical_Depth_055')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('PM2.5', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No PM2.5 data available for the specified parameters.'}), 404
                else:
                    PM2_5_mean = mean_image.multiply(206.91).add(41.181)
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                PM2_5_mean = filtered_collection.mean().clip(buffered_city_geometry) \
                    .multiply(206.91).add(41.181)

            pollutant_mean = PM2_5_mean.rename('PM2_5')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('PM2_5_min', None)
            max_value = stats.get('PM2_5_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for PM2.5.'}), 500

            base_unit = 'µg/m³'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('PM2_5_p5', None)
            max_value = percentiles.get('PM2_5_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for PM2.5.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = adjusted_unit

        elif pollutant == 'NO2':
            # Fetch and process the NO2 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('NO2_column_number_density')

            collection_size = filtered_collection.size().getInfo()
            print(f"NO2 collection size for the given parameters: {collection_size}")

            if collection_size == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('NO2', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No NO2 data available for the specified parameters.'}), 404
                else:
                    NO2_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                NO2_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = NO2_mean.rename('NO2')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=500,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('NO2_min', None)
            max_value = stats.get('NO2_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for NO2.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=500,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('NO2_p5', None)
            max_value = percentiles.get('NO2_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for NO2.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'NO2 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        elif pollutant == 'CO':
            # Fetch and process the CO data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select(['CO_column_number_density', 'H2O_column_number_density'])

            surface_pressure_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('surface_pressure')

            if filtered_collection.size().getInfo() == 0 or surface_pressure_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('CO', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No CO data available for the specified parameters.'}), 404
                else:
                    # We need H2O for CO calculation, try a larger approach for H2O too
                    filtered_collection_h2o = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                        .filterBounds(new_geom) \
                        .filterDate(start_date, end_date) \
                        .select(['H2O_column_number_density'])
                    if filtered_collection_h2o.size().getInfo() == 0:
                        return jsonify({'error': 'No CO data available for the specified parameters (H2O missing).'}), 404
                    def mask_negative_values(image):
                        return image.updateMask(image.gte(0))
                    filtered_collection_h2o = filtered_collection_h2o.map(mask_negative_values)
                    H2O_mean = filtered_collection_h2o.mean().clip(new_geom)

                    surface_pressure_collection_alt = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                        .filterBounds(new_geom) \
                        .filterDate(start_date, end_date) \
                        .select('surface_pressure')

                    if surface_pressure_collection_alt.size().getInfo() == 0:
                        return jsonify({'error': 'No CO data available for the specified parameters (Surface pressure missing).'}), 404
                    surface_pressure_mean = surface_pressure_collection_alt.mean().clip(new_geom)

                    CO_mean = mean_image
                    buffered_city_geometry = new_geom
                    # Calculate total column of dry air
                    TC_dry_air = surface_pressure_mean.divide(g * m_dry_air).subtract(H2O_mean.multiply(m_H2O / m_dry_air))
                    XCO_ppb = CO_mean.divide(TC_dry_air).multiply(1e9).rename('XCO_ppb')
                    pollutant_mean = XCO_ppb
                # end interpolation block
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)

                CO_mean = filtered_collection.select('CO_column_number_density').mean().clip(buffered_city_geometry)
                H2O_mean = filtered_collection.select('H2O_column_number_density').mean().clip(buffered_city_geometry)
                surface_pressure_mean = surface_pressure_collection.mean().clip(buffered_city_geometry)

                TC_dry_air = surface_pressure_mean.divide(g * m_dry_air).subtract(H2O_mean.multiply(m_H2O / m_dry_air))

                XCO_ppb = CO_mean.divide(TC_dry_air).multiply(1e9).rename('XCO_ppb')
                pollutant_mean = XCO_ppb

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('XCO_ppb_min', None)
            max_value = stats.get('XCO_ppb_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for CO.'}), 500

            base_unit = 'ppb'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('XCO_ppb_p5', None)
            max_value = percentiles.get('XCO_ppb_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for CO.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = adjusted_unit

        elif pollutant == 'SO2':
            # Fetch and process the SO2 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('SO2_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('SO2', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No SO2 data available for the specified parameters.'}), 404
                else:
                    SO2_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                SO2_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = SO2_mean.rename('SO2')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('SO2_min', None)
            max_value = stats.get('SO2_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for SO2.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('SO2_p5', None)
            max_value = percentiles.get('SO2_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for SO2.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'SO2 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        elif pollutant == 'O3':
            # Fetch and process the O3 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('O3_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('O3', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No O3 data available for the specified parameters.'}), 404
                else:
                    O3_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                O3_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = O3_mean.rename('O3')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('O3_min', None)
            max_value = stats.get('O3_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for O3.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('O3_p5', None)
            max_value = percentiles.get('O3_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for O3.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'O3 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        elif pollutant == 'HCHO':
            # Fetch and process the HCHO data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('tropospheric_HCHO_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('HCHO', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No HCHO data available for the specified parameters.'}), 404
                else:
                    HCHO_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                HCHO_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = HCHO_mean.rename('HCHO')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('HCHO_min', None)
            max_value = stats.get('HCHO_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for HCHO.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('HCHO_p5', None)
            max_value = percentiles.get('HCHO_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for HCHO.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'HCHO data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        else:
            return jsonify({'error': f"Unsupported pollutant: {pollutant}"}), 400

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        buffer_range = abs(max_value - min_value) * 0.1
        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red'],
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None

        map_id = pollutant_mean.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format

        min_value_sci = f"{min_value:.2e}"
        max_value_sci = f"{max_value:.2e}"

        return jsonify({
                        'tile_url': tile_url,
                        'min': min_value_sci,
                        'max': max_value_sci,
                        'min_raw': min_value,
                        'max_raw': max_value,
                        'unit': unit,
                        'legend_labels': legend_labels
                    })


    except Exception as e:
        return jsonify({'error': str(e)}), 500

POLLUTANT_CONFIGS = {
    'PM2.5': {
        'collection': 'MODIS/061/MCD19A2_GRANULES',
        'band': 'Optical_Depth_055',
        'scale_factor': 206.91,
        'offset': 41.181,
        'unit': 'kµg/m³'
    },
    'PM10': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_AER_AI',
        'band': 'absorbing_aerosol_index',
        'scale_factor': 50,  # Conversion factor for PM10
        'offset': 20,        # Offset for PM10
        'unit': 'μg/m³'
    },
    'NO2': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_NO2',
        'band': 'NO2_column_number_density',
        'unit': 'μmol/m²',
        'scale_factor':1000000,
        'offset':0,
    },
    'CO': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_CO',
        'band': 'CO_column_number_density',
        'unit': 'ppb',
        'scale_factor':3000,
        'offset':0
    },
    'SO2': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_SO2',
        'band': 'SO2_column_number_density',
        'unit': 'μmol/m²',
        'scale_factor':1000,
        'offset':0

    },
    'O3': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_O3',
        'band': 'O3_column_number_density',
        'unit': 'mol/m²'
    },
    'HCHO': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_HCHO',
        'band': 'tropospheric_HCHO_column_number_density',
        'scale_factor':100000,
        'offset':0,
        'unit': 'μmol/m²'

    }
}

@cache.memoize(timeout=3600)
def get_optimized_geometry(geojson_path, simplify_error=1000):
    """Load and optimize geometry from GeoJSON file."""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    geometry = ee.Geometry(geojson_data['features'][0]['geometry'])
    simplified_geom = geometry.simplify(maxError=simplify_error)
    bounds = simplified_geom.bounds()
    return {
        'geometry': simplified_geom,
        'bounds': bounds
    }

def process_pollutant_data(geometry_data, pollutant, start_date, end_date, scale=2000):
    """Process pollutant data for a given geometry with negative value masking."""
    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    geometry = geometry_data['geometry']

    # Initial collection filtering
    collection = ee.ImageCollection(config['collection']) \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .select(config['band'])

    # Add negative value masking
    def mask_negative_values(image):
        return image.updateMask(image.gte(0))
    
    collection = collection.map(mask_negative_values)

    # Calculate mean
    mean_image = collection.mean()
    
    # Apply scale factor and offset if specified
    if 'scale_factor' in config:
        mean_image = mean_image.multiply(config['scale_factor'])
        if 'offset' in config:
            mean_image = mean_image.add(config['offset'])

    # Create and apply mask
    mask = ee.Image.constant(1).clip(geometry).mask()
    
    masked_mean = mean_image.updateMask(mask).rename(pollutant)

    # Calculate statistics
    stats = masked_mean.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=bounds,
        scale=scale,
        maxPixels=1e8,
        bestEffort=True
    ).getInfo()

    return masked_mean, stats, config['unit']
@app.route('/api/get-pollutant-state', methods=['GET'])
def get_pollutant_state():
    try:
        # Extract parameters
        state = request.args.get('state')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')
        hml = request.args.get('hml', 'false').lower() == 'true'

        if not all([state, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get optimized state geometry
        state_data = get_optimized_geometry(f"flaskapp/static/state/{state}.geojson")
        
        # Process pollutant data
        masked_mean, stats, unit = process_pollutant_data(
            state_data, pollutant, start_date, end_date
        )

        # Extract min/max values
        min_value = stats.get(f'{pollutant}_p5', None)
        max_value = stats.get(f'{pollutant}_p95', None)

        if min_value is None or max_value is None:
            return jsonify({'error': f'Could not calculate data range for {pollutant}.'}), 500

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        # Set visualization parameters
        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red']
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None

        # Generate map
        map_id = masked_mean.getMapId(vis_params)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'min': f"{min_value:.2e}",
            'max': f"{max_value:.2e}",
            'min_raw': min_value,
            'max_raw': max_value,
            'unit': unit,
            'legend_labels': legend_labels
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-pollutant-city', methods=['GET'])
def get_pollutant_city():
    try:
        # Extract parameters
        city = request.args.get('city')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')
        hml = request.args.get('hml', 'false').lower() == 'true'

        if not all([city, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get optimized city geometry with smaller simplification error
        city_data = get_optimized_geometry(
            f"flaskapp/static/dissolved_output/dissolved_{city.upper()}.geojson",
            simplify_error=100  # Smaller error for city boundaries
        )
        
        # Process pollutant data with higher resolution for cities
        masked_mean, stats, unit = process_pollutant_data(
            city_data, pollutant, start_date, end_date, scale=1000  # Higher resolution for cities
        )

        # Extract min/max values
        min_value = stats.get(f'{pollutant}_p5', None)
        max_value = stats.get(f'{pollutant}_p95', None)

        if min_value is None or max_value is None:
            return jsonify({'error': f'Could not calculate data range for {pollutant}.'}), 500

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        # Set visualization parameters
        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red']
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None

        # Generate map
        map_id = masked_mean.getMapId(vis_params)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'min': f"{min_value:.2e}",
            'max': f"{max_value:.2e}",
            'min_raw': min_value,
            'max_raw': max_value,
            'unit': unit,
            'legend_labels': legend_labels
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/get-time-series', methods=['GET'])
def get_time_series():
    try:
        # Extract and validate input parameters
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        pollutant = request.args.get('pollutant')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not all([lat, lon, pollutant, start_date, end_date]):
            return jsonify({'error': 'Missing required parameters.'}), 400

        if pollutant not in POLLUTANT_CONFIGS:
            return jsonify({'error': f'Invalid pollutant: {pollutant}'}), 400

        # Create a point geometry with a small buffer to ensure data capture
        point = ee.Geometry.Point([lon, lat]).buffer(1000)

        # Get the pollutant configuration
        config = POLLUTANT_CONFIGS[pollutant]

        # Create the base collection
        collection = ee.ImageCollection(config['collection'])\
            .filterDate(start_date, end_date)\
            .filterBounds(point)\
            .select(config['band'])

        # Add negative value masking
        def mask_negative_values(image):
            return image.updateMask(image.gte(0))
        
        collection = collection.map(mask_negative_values)

        # Debug: Log the collection size
        collection_size = collection.size().getInfo()
        print(f"Collection size: {collection_size}")

        if collection_size == 0:
            return jsonify({'error': 'No data available for the specified dates.'}), 404

        # Get all unique dates in the collection
        def get_image_date(image):
            date = ee.Date(image.get('system:time_start'))
            return ee.Feature(None, {'date': date.format('YYYY-MM-dd')})

        dates = collection.map(get_image_date).distinct('date').aggregate_array('date').getInfo()
        
        # Process each image in the collection
        def process_image(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            value = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000,
                maxPixels=1e9
            ).get(config['band'])
            
            return ee.Feature(None, {
                'date': date,
                'value': value
            })

        features = collection.map(process_image)
        time_series_data = features.getInfo().get('features', [])

        # Process the data and apply scaling factors
        series_data = []
        for feature in time_series_data:
            props = feature['properties']
            value = props.get('value')
            
            if value is not None and not isinstance(value, str):  # Ensure value is numeric and not None
                if value >= 0:  # Additional check for negative values
                    if 'scale_factor' in config:
                        value = value * config['scale_factor'] + config.get('offset', 0)
                    
                    # Round the value to 4 decimal places for cleaner data
                    value = round(float(value), 4)
                    
                    series_data.append({
                        'date': props['date'],
                        'value': value
                    })

        # Sort the data by date
        series_data.sort(key=lambda x: x['date'])

        # Remove duplicates while keeping the first occurrence
        seen_dates = set()
        unique_series_data = []
        for data_point in series_data:
            if data_point['date'] not in seen_dates:
                seen_dates.add(data_point['date'])
                unique_series_data.append(data_point)

        return jsonify({
            'series': unique_series_data,
            'unit': config.get('unit', 'unknown'),
            'dates': dates
        })

    except ee.EEException as gee_error:
        print(f"GEE Error: {str(gee_error)}")
        return jsonify({'error': f'Google Earth Engine error: {str(gee_error)}'}), 500
    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# API route to get the Windy API key
@app.route('/api/get-windy-api-key', methods=['GET'])
def get_windy_api_key():
    if WINDY_API_KEY:
        return jsonify({'api_key': WINDY_API_KEY})
    else:
        return jsonify({'error': 'Windy API key not configured.'}), 500


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class AirQualityChatbot:
    def __init__(self, app):
        self.app = app
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.lemmatizer = lambda x: x  # Simple pass-through function
            self.stop_words = set()
            print("Warning: NLTK resources not fully available.")
        
        self.pollutant_info = {
            'SO2': 'Sulfur dioxide (SO2) is a toxic gas with a pungent odor. It\'s primarily produced from the burning of fossil fuels containing sulfur.',
            'NO2': 'Nitrogen dioxide (NO2) is a reddish-brown gas that primarily comes from the burning of fuel.',
            'CO': 'Carbon monoxide (CO) is a colorless, odorless gas that\'s produced by incomplete combustion of carbon-based fuels.',
            'O3': 'Ozone (O3) at ground level is a harmful air pollutant and a key component of smog.',
            'PM2.5': 'PM2.5 refers to fine particulate matter smaller than 2.5 micrometers in diameter.',
            'PM10': 'PM10 refers to particulate matter up to 10 micrometers in size.',
            'HCHO': 'Formaldehyde (HCHO) is a colorless gas that can cause irritation to the eyes, nose, and throat.'
        }

        self.city_coordinates = {
            'hyderabad': {'lat': 17.3850, 'lon': 78.4867},
            'srinagar': {'lat': 34.0837, 'lon': 74.7973},
            'delhi': {'lat': 28.6139, 'lon': 77.2090},
            'mumbai': {'lat': 19.0760, 'lon': 72.8777},
            'bangalore': {'lat': 12.9716, 'lon': 77.5946}
        }

    def preprocess_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return tokens
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return text.lower().split()

    def extract_date_range(self, text):
        date_patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4})',
            r'(\d{2}/\d{2}/\d{4})',  # e.g., 01/01/2024
            r'(\d{4}-\d{2}-\d{2})'   # e.g., 2024-01-01
        ]
        
        dates = []
        for pattern in date_patterns:
            found_dates = re.findall(pattern, text.lower())
            dates.extend(found_dates)
        
        if len(dates) >= 2:
            try:
                if "/" in dates[0]:
                    start_date = datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
                else:
                    start_date = datetime.strptime(dates[0], '%d %B %Y').strftime('%Y-%m-%d')
                
                if "/" in dates[1]:
                    end_date = datetime.strptime(dates[1], '%d/%m/%Y').strftime('%Y-%m-%d')
                else:
                    end_date = datetime.strptime(dates[1], '%d %B %Y').strftime('%Y-%m-%d')

                return start_date, end_date
            except Exception as e:
                print(f"Error in date conversion: {e}")
        return None, None

    def extract_city(self, text):
        tokens = self.preprocess_text(text)
        for city in self.city_coordinates.keys():
            if city in tokens:
                return city
        return None

    def extract_pollutant(self, text):
        tokens = self.preprocess_text(text)
        for pollutant in self.pollutant_info.keys():
            if pollutant.lower() in tokens:
                return pollutant
        return None

    def get_pollutant_data(self, city, pollutant, start_date, end_date):
        """Fetch pollutant data from an API endpoint."""
        with self.app.test_client() as client:
            city_coords = self.city_coordinates[city]
            response = client.get(f'/api/get-pollutant?lat={city_coords["lat"]}&lon={city_coords["lon"]}&pollutant={pollutant}&start_date={start_date}&end_date={end_date}')
            return response.get_json()

    def get_pollutant_stats(self, city, pollutant, start_date, end_date):
        """Get statistical information about pollutant levels."""
        data = self.get_pollutant_data(city, pollutant, start_date, end_date)
        if 'error' in data:
            return f"Sorry, I couldn't retrieve the data: {data['error']}"
        
        min_val = float(data['min'])
        max_val = float(data['max'])
        unit = data['unit']
        
        return {
            'min': min_val,
            'max': max_val,
            'unit': unit,
            'average': (min_val + max_val) / 2
        }

    def generate_response(self, user_input):
        city = self.extract_city(user_input)
        pollutant = self.extract_pollutant(user_input)
        start_date, end_date = self.extract_date_range(user_input)

        if 'what is' in user_input.lower() and pollutant and not city:
            return self.pollutant_info.get(pollutant, "I don't have information about that pollutant.")

        if city and pollutant and start_date and end_date:
            try:
                stats = self.get_pollutant_stats(city, pollutant, start_date, end_date)
                response = f"For {city.title()} between {start_date} and {end_date}:\n"
                response += f"The {pollutant} levels ranged from {stats['min']:.2f} to {stats['max']:.2f} {stats['unit']}\n"
                response += f"The average concentration was approximately {stats['average']:.2f} {stats['unit']}"
                return response
            except Exception as e:
                return f"I apologize, but I encountered an error while retrieving the data: {str(e)}"

        missing_info = []
        if not city:
            missing_info.append("city")
        if not pollutant:
            missing_info.append("pollutant type")
        if not start_date or not end_date:
            missing_info.append("date range")
        
        if missing_info:
            return f"I need more information to answer your question. Please specify the {', '.join(missing_info)}."

        return "I'm not sure how to help with that query. Please try asking about specific pollutant levels in a city for a particular date range."

# Route to chat with chatbot
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        chatbot = AirQualityChatbot(app)
        response = chatbot.generate_response(user_message)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)