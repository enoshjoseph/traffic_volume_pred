import streamlit as st
import pickle
import pandas as pd
import requests
import xgboost as xgb
import datetime

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="Traffic Prediction", page_icon="🚦", layout="wide")

# --------------------------
# Load Trained Model
# --------------------------
# Define the fixed location parameters for Ljubljana, Slovenia
FIXED_LAT = 46.0569
FIXED_LON = 14.5058
FIXED_CITY_NAME = "Ljubljana"
FIXED_COUNTRY = "SI"

@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        model.load_model("xgb_traffic_model.json")
        return model
    except xgb.core.XGBoostError as e:
        st.error(f"❌ Could not load XGBoost model: {e}")
        st.info("💡 Ensure 'xgb_traffic_model.json' exists and is a valid XGBoost model file.")
        return None

model = load_model()

# --------------------------
# App Title
# --------------------------
st.title(f"🚦 Traffic Volume Prediction for {FIXED_CITY_NAME}")
st.markdown("---")

# --------------------------
# Sidebar for Configuration (Simplified)
# --------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Removed: st.text_input("Enter City Name", "Ljubljana")
    # Removed: use_coords checkbox
    
    st.markdown("---")
    st.info(f"📍 **Fixed Prediction Location:**\n**{FIXED_CITY_NAME}, Slovenia**\nLat: {FIXED_LAT}, Lon: {FIXED_LON}\n\nThis app is specifically trained for this region.")

# --------------------------
# OpenWeather API Configuration (Hardcoded for Ljubljana Coordinates)
# --------------------------
try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except KeyError:
    st.error("🔑 API Key not found! Please set `OPENWEATHER_API_KEY` in your Streamlit secrets.")
    API_KEY = None

# Build URL using the fixed coordinates
if API_KEY:
    URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={FIXED_LAT}&lon={FIXED_LON}&appid={API_KEY}&units=metric"
else:
    URL = None 

# --------------------------
# Prediction Button
# --------------------------
if st.button("🔍 Predict Tomorrow's Traffic", type="primary"):
    if model is None:
        st.error("Cannot make predictions without a trained model.")
    elif URL is None:
        st.warning("⚠️ Cannot fetch weather data without a valid OpenWeather API key.")
    else:
        with st.spinner("Fetching weather data..."):
            try:
                response = requests.get(URL, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    # Use fixed location names since we know the coordinates
                    location_name = FIXED_CITY_NAME
                    country = FIXED_COUNTRY 
                    st.success(f"✅ Fetched 5-day forecast data for **{location_name}, {country}**")

                    # --------------------------
                    # Extract forecast data for tomorrow
                    # --------------------------
                    forecast_list = []
                    today = datetime.datetime.now().date()
                    tomorrow = today + datetime.timedelta(days=1)
                    
                    for entry in data["list"]:
                        dt_txt = entry["dt_txt"]
                        date_time = datetime.datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")

                        # Get only tomorrow's data
                        if date_time.date() == tomorrow:
                            temp = entry["main"]["temp"]
                            humidity = entry["main"]["humidity"]
                            pressure = entry["main"]["pressure"]
                            wind_speed = entry["wind"]["speed"]
                            weather = entry["weather"][0]["main"]
                            weather_desc = entry["weather"][0]["description"]
                            
                            # Add rain volume if present, otherwise default to 0
                            rain_1h = entry.get("rain", {}).get("3h", 0) / 3 
                            
                            # Additional features
                            feels_like = entry["main"]["feels_like"]
                            clouds = entry["clouds"]["all"]

                            forecast_list.append({
                                "datetime": date_time,
                                "temp": temp,
                                "feels_like": feels_like,
                                "humidity": humidity,
                                "pressure": pressure,
                                "wind_speed": wind_speed,
                                "clouds": clouds,
                                "weather": weather,
                                "weather_desc": weather_desc,
                                "rain_1h": rain_1h
                            })

                    # --------------------------
                    # Convert to DataFrame
                    # --------------------------
                    if not forecast_list:
                        st.warning("⚠️ No forecast data available for tomorrow. Try again later.")
                    else:
                        df_forecast = pd.DataFrame(forecast_list)
                        
                        # Display weather forecast
                        st.subheader("🌤 Tomorrow's Weather Forecast")
                        st.dataframe(
                            df_forecast[["datetime", "temp", "humidity", "pressure", "wind_speed", "weather_desc"]],
                            use_container_width=True
                        )

                        # --------------------------
                        # Feature Engineering
                        # --------------------------
                        df_features = df_forecast.copy()
                        
                        # Extract datetime features
                        df_features["hour"] = df_features["datetime"].dt.hour
                        df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
                        df_features["month"] = df_features["datetime"].dt.month
                        
                        # Use actual or imputed rain_1h (already handled in extraction)
                        # df_features["rain_1h"] = 0.0 # Keeping the original line for safety if extraction failed

                        # Add is_holiday (assuming False for predictions)
                        df_features["is_holiday_True"] = 0
                        
                        # Rename clouds to clouds_all
                        df_features["clouds_all"] = df_features["clouds"]
                        
                        # Create one-hot encoded features for weather_main
                        weather_main_categories = [
                            'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 
                            'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'
                        ]
                        for category in weather_main_categories:
                            df_features[f"weather_main_{category}"] = (df_features["weather"] == category).astype(int)
                        
                        # Create one-hot encoded features for weather_description
                        weather_desc_categories = [
                            'Sky is Clear', 'broken clouds', 'drizzle', 'few clouds', 
                            'fog', 'freezing rain', 'haze', 'heavy intensity drizzle',
                            'heavy intensity rain', 'heavy snow', 'light intensity drizzle',
                            'light intensity shower rain', 'light rain', 'light rain and snow',
                            'light shower snow', 'light snow', 'mist', 'moderate rain',
                            'overcast clouds', 'proximity shower rain', 'proximity thunderstorm',
                            'proximity thunderstorm with drizzle', 'proximity thunderstorm with rain',
                            'scattered clouds', 'shower drizzle', 'shower snow', 'sky is clear',
                            'sleet', 'smoke', 'snow', 'thunderstorm', 'thunderstorm with drizzle',
                            'thunderstorm with heavy rain', 'thunderstorm with light drizzle',
                            'thunderstorm with light rain', 'thunderstorm with rain',
                            'very heavy rain'
                        ]
                        for category in weather_desc_categories:
                            df_features[f"weather_description_{category}"] = (df_features["weather_desc"] == category).astype(int)
                        
                        # Define all expected features in correct order (EXCLUDING 'date_time' and 'traffic_volume')
                        expected_features = [
                            'temp', 'rain_1h', 'clouds_all', 
                             'day_of_week', 'hour', 'month',
                            'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze',
                            'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
                            'weather_main_Squall', 'weather_main_Thunderstorm', 'weather_description_Sky is Clear', 'weather_description_broken clouds',
                            'weather_description_drizzle', 'weather_description_few clouds', 'weather_description_fog', 'weather_description_freezing rain',
                            'weather_description_haze', 'weather_description_heavy intensity drizzle', 'weather_description_heavy intensity rain', 'weather_description_heavy snow',
                            'weather_description_light intensity drizzle', 'weather_description_light intensity shower rain', 'weather_description_light rain', 'weather_description_light rain and snow',
                            'weather_description_light shower snow', 'weather_description_light snow', 'weather_description_mist', 'weather_description_moderate rain',
                            'weather_description_overcast clouds', 'weather_description_proximity shower rain', 'weather_description_proximity thunderstorm', 'weather_description_proximity thunderstorm with drizzle',
                            'weather_description_proximity thunderstorm with rain', 'weather_description_scattered clouds', 'weather_description_shower drizzle', 'weather_description_shower snow',
                            'weather_description_sky is clear', 'weather_description_sleet', 'weather_description_smoke', 'weather_description_snow',
                            'weather_description_thunderstorm', 'weather_description_thunderstorm with drizzle', 'weather_description_thunderstorm with heavy rain', 'weather_description_thunderstorm with light drizzle',
                            'weather_description_thunderstorm with light rain', 'weather_description_thunderstorm with rain', 'weather_description_very heavy rain', 'is_holiday_True'
                        ]
                        
                        # Ensure all expected features exist (add missing ones with 0)
                        for feature in expected_features:
                            if feature not in df_features.columns:
                                df_features[feature] = 0
                        
                        # Select features in the correct order
                        X = df_features[expected_features]

                        # --------------------------
                        # Predict Traffic Volume
                        # --------------------------
                        try:
                            # 1. Convert the Pandas DataFrame X into an XGBoost DMatrix
                            dmatrix_X = xgb.DMatrix(X, feature_names=X.columns.tolist())
                            
                            # 2. Use the DMatrix for prediction
                            predictions = model.predict(dmatrix_X)
                            
                            # 3. Store results and continue
                            df_forecast["Predicted_Traffic_Volume"] = predictions.astype(int)

                            st.subheader("🚗 Predicted Traffic Volume for Tomorrow")
                                
                            # Display predictions with styling
                            display_df = df_forecast[["datetime", "weather", "temp", "Predicted_Traffic_Volume"]].copy()
                            display_df["datetime"] = display_df["datetime"].dt.strftime("%I:%M %p")
                                
                            st.dataframe(display_df, use_container_width=True)

                            # --------------------------
                            # Statistics
                            # --------------------------
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🚗 Average Traffic", f"{predictions.mean():.0f}")
                            with col2:
                                st.metric("📈 Peak Traffic", f"{predictions.max():.0f}")
                            with col3:
                                st.metric("📉 Minimum Traffic", f"{predictions.min():.0f}")
                            with col4:
                                peak_index = predictions.argmax()
                                peak_hour = df_forecast.loc[peak_index, "datetime"].strftime("%I:%M %p")
                                st.metric("⏰ Peak Hour", peak_hour)

                            # --------------------------
                            # Visualization
                            # --------------------------
                            st.subheader("📊 Traffic Volume Throughout Tomorrow")
                                
                            # Prepare data for chart
                            chart_data = df_forecast.copy()
                            chart_data["Time"] = chart_data["datetime"].dt.strftime("%I:%M %p")
                            chart_data = chart_data.set_index("Time")
                                
                            st.line_chart(chart_data["Predicted_Traffic_Volume"], height=400)
                                
                            # --------------------------
                            # Download Option
                            # --------------------------
                            csv = df_forecast.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name=f"traffic_predictions_{tomorrow}.csv",
                                mime="text/csv"
                            )
                                
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
                            st.info("💡 Make sure your model was trained with the same features.")
                        
                elif response.status_code == 404:
                    # This should rarely happen now since the city is hardcoded by coordinates
                    st.error(f"❌ Forecast data not found for {FIXED_CITY_NAME}. OpenWeather may have changed API or the location is unavailable.")
                elif response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your OpenWeather API key.")
                else:
                    st.error(f"❌ Failed to fetch weather data. Status code: {response.status_code}")
                        
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.RequestException as e:
                # Retained network error handling
                st.error(f"❌ Network error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Prediction for Ljubljana, Slovenia | Powered by OpenWeather API 🌦️</p>
    </div>
    """,
    unsafe_allow_html=True
)