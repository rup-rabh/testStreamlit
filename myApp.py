import streamlit as st
import pandas as pd
import ee
import geemap.foliumap as geemap
import numpy as np

# Authenticate and initialize Earth Engine
try:
    ee.Initialize(project='eegeemaps')
except:
    ee.Authenticate()
    ee.Initialize(project='eegeemaps')

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Airport Analysis System")

# Load airports data (cached)
@st.cache_data
def load_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    columns = [
        "id", "name", "city", "country", "iata", "icao", "lat", "lon", "altitude",
        "timezone", "dst", "tz_db", "type", "source"
    ]
    df = pd.read_csv(url, header=None, names=columns)
    return df

def get_lat_lon_from_iata(iata_code, airports_df):
    """Get latitude and longitude from IATA code"""
    airport_row = airports_df[airports_df["iata"] == iata_code.upper()]
    if len(airport_row) > 0:
        lat = float(airport_row.iloc[0]["lat"])
        lon = float(airport_row.iloc[0]["lon"])
        return lat, lon, True
    return 0.0, 0.0, False

def generate_thresholds(min_val, max_val, steps=10):
    """Generate threshold values between min and max"""
    step_size = (max_val - min_val) / steps
    return [round(min_val + i * step_size, 3) for i in range(steps + 1)]

def get_threshold_configs():
    """Define threshold configurations for different indices"""
    return {
        'NDVI': generate_thresholds(-1, 1, 10),
        'NDWI': generate_thresholds(-1, 1, 10),
        'EVI': generate_thresholds(-1, 1, 10),
        'MNDWI': generate_thresholds(-1, 0.5, 10),
        'NDII': generate_thresholds(-1, 1, 10)
    }

def get_color_palettes():
    """Define color palettes for visualization"""
    return {
        'NDVI': ['#808080', '#654321', '#b58900', '#fdfd96', '#adff2f', 
                '#7fff00', '#32cd32', '#228b22', '#006400', '#004d00'],
        'NDWI': ['#f0f8ff','#d4f1f9', '#a2dff7', '#73c2fb', '#3fa9f5', 
                '#1e90ff', '#0077be', '#005f8d', '#003f5c', '#001f3f'],
        'EVI': ['#808080', '#ff7f00', '#ffb347', '#ffff66', '#ccff99', 
               '#99ff99', '#66ff66', '#32cd32', '#228b22', '#006400'],
        'MNDWI': ['#00008b','#0000cd','#4169e1','#4682b4','#87ceeb',
                 '#b0e0e6','#f5f5dc','#f0f0f0','#ffffff', '#fffff0'],
        'NDII': ['#d9f0ff','#ccece6','#b3e2cd','#a2d9c3','#90d0b9',
                '#7fcdbb','#66c2a5','#41b6c4','#1d91c0','#0b6283']
    }

def compute_indices(image):
    """Compute vegetation and water indices"""
    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDWI: (Green - NIR) / (Green + NIR)
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # EVI: Enhanced Vegetation Index
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    
    # MNDWI: Modified Normalized Difference Water Index
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    
    # NDII: Normalized Difference Infrared Index
    ndii = image.normalizedDifference(['B8', 'B11']).rename('NDII')
    
    return image.addBands([ndvi, ndwi, evi, mndwi, ndii])

def get_sentinel2_data(lat, lon, start_date, end_date, buffer_km=2.735):
    """Get Sentinel-2 data for the specified location and date range"""
    center = ee.Geometry.Point([lon, lat])
    buffer_radius = (buffer_km / 2) * 1000  # Convert km to meters
    bounds = center.buffer(buffer_radius).bounds()
    
    # Get Sentinel-2 Surface Reflectance data
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .median() \
        .multiply(0.0001) \
        .clip(bounds)
    
    # Compute indices
    image = compute_indices(image)
    
    return image, bounds, center

def get_landsat9_temperature(lat, lon, start_date, end_date, buffer_km=2.735):
    """Get Landsat 9 temperature data"""
    center = ee.Geometry.Point([lon, lat])
    buffer_radius = (buffer_km / 2) * 1000
    bounds = center.buffer(buffer_radius).bounds()
    
    # Cloud masking function
    def mask_clouds(img):
        cloud_mask = 1 << 3
        cloud_shadow_mask = 1 << 4
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(cloud_mask).eq(0).And(
               qa.bitwiseAnd(cloud_shadow_mask).eq(0))
        return img.updateMask(mask)
    
    # Scale function
    def apply_scale(img):
        optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = img.select('ST_B.*').multiply(0.00341802).add(149.0)
        return img.addBands(optical, None, True).addBands(thermal, None, True)
    
    # Temperature conversion
    def add_temp_c(img):
        temp_k = img.select('ST_B10')
        temp_c = temp_k.subtract(273.15).rename('temp_C')
        return img.addBands(temp_c)
    
    # Get Landsat 9 data
    temp_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .map(mask_clouds) \
        .map(apply_scale) \
        .map(add_temp_c)
    
    temp_mean = temp_collection.select('temp_C').mean().clip(bounds)
    
    return temp_mean

def enhanced_vegetation_analysis_tab():
    """Enhanced vegetation analysis with multiple indices and temperature"""
    with st.sidebar:
        st.header("Enhanced Vegetation Analysis")
        
        # Load airports
        airports_df = load_airports()
        
        # Airport selection (dropdown with IATA codes)
        airport_options = [f"{row['name']} ({row['iata']})" 
                          for _, row in airports_df.dropna(subset=['iata']).iterrows()]
        selected_airport_display = st.selectbox(
            "Select Airport",
            options=airport_options,
            index=0,
            key="enhanced_veg_airport"
        )
        
        # Extract IATA code
        iata_code = selected_airport_display.split('(')[1].split(')')[0]
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-06-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-31"))
        
        # Index selection
        selected_indices = st.multiselect(
            "Select Indices to Display",
            options=['NDVI', 'NDWI', 'EVI', 'MNDWI', 'NDII'],
            default=['NDVI', 'NDWI'],
            key="selected_indices"
        )
        
        # Threshold selections
        st.subheader("Threshold Settings")
        thresholds = get_threshold_configs()
        threshold_values = {}
        
        for index in selected_indices:
            threshold_values[index] = st.slider(
                f"{index} Threshold",
                min_value=float(thresholds[index][0]),
                max_value=float(thresholds[index][-1]),
                value=0.5 if index != 'MNDWI' else 0.1,
                step=0.01,
                key=f"{index}_threshold"
            )
        
        # Temperature analysis toggle
        show_temperature = st.checkbox("Include Temperature Analysis", value=True)
        
        # Buffer size
        buffer_size = st.slider(
            "Analysis Buffer (km)",
            min_value=1.0,
            max_value=5.0,
            value=2.735,
            step=0.1
        )
        
        # Analysis button
        run_analysis = st.button("Run Enhanced Analysis", type="primary")
        
        # Display parameters
        st.markdown("---")
        st.subheader("Analysis Parameters:")
        st.write(f"**Airport:** {selected_airport_display}")
        st.write(f"**Date Range:** {start_date} to {end_date}")
        st.write(f"**Indices:** {', '.join(selected_indices)}")
        st.write(f"**Buffer:** {buffer_size} km")

    # Main analysis area
    if run_analysis:
        try:
            # Get airport coordinates
            lat, lon, found = get_lat_lon_from_iata(iata_code, airports_df)
            
            if not found:
                st.error(f"Airport with IATA code '{iata_code}' not found!")
                return
            
            st.success(f"Analyzing {selected_airport_display} at coordinates: {lat:.4f}, {lon:.4f}")
            
            with st.spinner("Processing satellite data..."):
                # Get Sentinel-2 data
                image, bounds, center = get_sentinel2_data(
                    lat, lon, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d'),
                    buffer_size
                )
                
                # Create map
                Map = geemap.Map(center=[lat, lon], zoom=13)
                
                # Add boundary and center
                Map.addLayer(bounds, {'color': 'FF0000', 'fillOpacity': 0.1}, 'Analysis Area')
                Map.addLayer(center, {'color': '0000FF'}, 'Airport Center')
                
                # Get color palettes
                palettes = get_color_palettes()
                
                # Add selected indices to map
                for index in selected_indices:
                    if index in palettes:
                        vis_params = {
                            'min': thresholds[index][0],
                            'max': thresholds[index][-1],
                            'palette': palettes[index]
                        }
                        Map.addLayer(image.select(index), vis_params, f'{index} Visualization')
                        
                        # Add threshold highlighting
                        if index in threshold_values:
                            threshold_mask = image.select(index).gt(threshold_values[index])
                            Map.addLayer(
                                threshold_mask.updateMask(threshold_mask),
                                {'palette': ['darkgreen']},
                                f'{index} > {threshold_values[index]}'
                            )
                
                # Add temperature data if requested
                if show_temperature:
                    with st.spinner("Processing temperature data..."):
                        temp_data = get_landsat9_temperature(
                            lat, lon,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'),
                            buffer_size
                        )
                        
                        temp_vis = {
                            'min': 10,
                            'max': 30,
                            'palette': ['blue', 'green', 'red']
                        }
                        Map.addLayer(temp_data, temp_vis, 'Mean Temperature (°C)')
                        Map.add_colorbar(temp_vis, label="Temperature (°C)")
                
                # Display map
                Map.to_streamlit(height=700)
                
                # Summary statistics
                st.subheader("Analysis Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Analysis Area", f"{buffer_size} km radius")
                    st.metric("Date Range", f"{(end_date - start_date).days} days")
                
                with col2:
                    st.metric("Indices Analyzed", len(selected_indices))
                    if show_temperature:
                        st.metric("Temperature Analysis", "✅ Included")
                
                with col3:
                    st.metric("Airport Code", iata_code)
                    st.metric("Coordinates", f"{lat:.3f}, {lon:.3f}")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Please check your Earth Engine authentication and try again.")
    
    else:
        st.info("👆 Configure your analysis parameters and click 'Run Enhanced Analysis' to begin.")

def runway_analysis_tab():
    """Existing runway analysis tab (unchanged)"""
    with st.sidebar:
        st.header("Runway Crack Analysis Controls")
        airports_df = load_airports()
        selected_airport = st.selectbox(
            "Select Airport",
            options=airports_df["name"].tolist(),
            index=0,
            key="runway_airport"
        )
        st.markdown("---")
        st.subheader("Selected Parameters:")
        st.write(f"Airport: {selected_airport}")

    st.header("Runway Crack Analysis")
    st.info("Runway crack analysis functionality coming soon!")

def main():
    st.title("🛩️ Enhanced Airport Analysis System")
    st.markdown("Advanced satellite-based analysis for airport vegetation and infrastructure monitoring")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🌱 Enhanced Vegetation Analysis", "🛣️ Runway Analysis"])
    
    with tab1:
        enhanced_vegetation_analysis_tab()
    
    with tab2:
        runway_analysis_tab()

if __name__ == "__main__":
    main()
