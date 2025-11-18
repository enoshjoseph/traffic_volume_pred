# import streamlit as st
# import pickle
# import pandas as pd
# import requests
# import xgboost as xgb
# import datetime
# import numpy as np

# # Suppress warnings related to setting values on DataFrame copies or scikit-learn feature names
# import warnings
# warnings.filterwarnings("ignore")

# # --------------------------
# # Global Configuration & Feature Lists
# # --------------------------
# FIXED_LAT = 46.0569
# FIXED_LON = 14.5058
# FIXED_CITY_NAME = "Ljubljana"
# FIXED_COUNTRY = "SI"
# # Use st.secrets to safely load the key
# API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE") 

# # --- Feature Lists (MUST match training structure) ---
# TRAINING_FEATURES = [
#     'temp', 'rain_1h', 'clouds_all', 'day_of_week', 'hour', 'month',
#     'traffic_volume_lag1', 'traffic_volume_lag24',
#     'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze',
#     'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
#     'weather_main_Squall', 'weather_main_Thunderstorm', 'weather_description_Sky is Clear', 
#     'weather_description_broken clouds', 'weather_description_drizzle', 'weather_description_few clouds', 
#     'weather_description_fog', 'weather_description_freezing rain', 'weather_description_haze', 
#     'weather_description_heavy intensity drizzle', 'weather_description_heavy intensity rain', 
#     'weather_description_heavy snow', 'weather_description_light intensity drizzle', 
#     'weather_description_light intensity shower rain', 'weather_description_light rain', 
#     'weather_description_light rain and snow', 'weather_description_light shower snow', 
#     'weather_description_light snow', 'weather_description_mist', 'weather_description_moderate rain', 
#     'weather_description_overcast clouds', 'weather_description_proximity shower rain', 
#     'weather_description_proximity thunderstorm', 'weather_description_proximity thunderstorm with drizzle',
#     'weather_description_proximity thunderstorm with rain', 'weather_description_scattered clouds', 
#     'weather_description_shower drizzle', 'weather_description_shower snow', 'weather_description_sky is clear', 
#     'weather_description_sleet', 'weather_description_smoke', 'weather_description_snow', 
#     'weather_description_thunderstorm', 'weather_description_thunderstorm with drizzle', 
#     'weather_description_thunderstorm with heavy rain', 'weather_description_thunderstorm with light drizzle',
#     'weather_description_thunderstorm with light rain', 'weather_description_thunderstorm with rain',
#     'weather_description_very heavy rain', 'is_holiday_True'
# ]

# NUMERIC_COLS_TO_SCALE = [
#     'temp', 'rain_1h', 'clouds_all', 'hour', 'day_of_week', 'month',
#     'traffic_volume_lag1', 'traffic_volume_lag24'
# ]

# HOURLY_MULTIPLIERS = {
#     0: 0.37, 1: 0.25, 2: 0.18, 3: 0.15, 4: 0.20, 5: 0.47,
#     6: 1.08, 7: 1.58, 8: 1.77, 9: 1.47, 10: 1.27, 11: 1.22,
#     12: 1.27, 13: 1.37, 14: 1.47, 15: 1.67, 16: 1.88, 17: 1.97,
#     18: 1.77, 19: 1.37, 20: 0.97, 21: 0.77, 22: 0.62, 23: 0.47
# }

# # --------------------------
# # Load Trained Model and Scaler
# # --------------------------

# @st.cache_resource
# def load_assets():
#     """Loads the model and scaler and extracts necessary stats."""
#     try:
#         model = xgb.Booster()
#         model.load_model("xgb_traffic_modell.json")
        
#         with open("scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
        
#         lag1_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag1')
#         traffic_mean = scaler.mean_[lag1_idx]
#         traffic_std = scaler.scale_[lag1_idx]
#         lag24_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag24')
        
#         return model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx
#     except xgb.core.XGBoostError as e:
#         st.error(f"❌ Could not load XGBoost model: {e}")
#         st.info("💡 Ensure 'xgb_traffic_modell.json' exists and is a valid XGBoost model file.")
#         return None, None, None, None, None, None
#     except Exception as e:
#         # Added a safe fallback for the TypeError in the sidebar
#         return None, None, None, None, None, None

# model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx = load_assets()

# # --------------------------
# # Main Prediction Function
# # --------------------------

# def get_hourly_baseline(traffic_mean):
#     """Calculates the hourly baseline traffic volumes."""
#     return {hour: traffic_mean * HOURLY_MULTIPLIERS.get(hour, 1.0) for hour in range(24)}

# def run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx):
    
#     df_features = df_weather_data.copy()
    
#     # 1. Feature Engineering
#     df_features["hour"] = df_features["datetime"].dt.hour
#     df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
#     df_features["month"] = df_features["datetime"].dt.month
#     df_features["clouds_all"] = df_features["clouds"]
#     df_features["is_holiday_True"] = 0
#     df_features['weather'] = df_features['weather_main']
#     df_features['weather_desc'] = df_features['weather_description'] 
    
#     df_encoded = pd.get_dummies(df_features, columns=["weather", "weather_desc"],  
#                                 prefix=["weather_main", "weather_description"], dtype=int) 
    
#     baseline_traffic = get_hourly_baseline(traffic_mean)
    
#     # 2. Prediction Loop (Pure Cascading)
#     predicted_volumes = [] 
#     previous_predicted_volume = baseline_traffic[23] # Start with 23:00 baseline

#     for i in range(len(df_encoded)): 
#         row = df_encoded.iloc[[i]].copy()
#         current_hour = row['hour'].values[0] 
        
#         # Lag Feature Assignment (Pure Cascading)
#         row["traffic_volume_lag1"] = previous_predicted_volume
#         row["traffic_volume_lag24"] = baseline_traffic[current_hour] 
        
#         # Align Features
#         X_pred_row = pd.DataFrame(0, index=[0], columns=TRAINING_FEATURES) 
#         for col in X_pred_row.columns: 
#             if col in row.columns: 
#                 X_pred_row[col] = row[col].values[0] 
        
#         # --- SCALING FIX: The critical 8-feature scaling ---
#         X_numeric = X_pred_row[NUMERIC_COLS_TO_SCALE].copy()
#         lag1_unscaled = X_numeric['traffic_volume_lag1'].values[0]
#         lag24_unscaled = X_numeric['traffic_volume_lag24'].values[0]
        
#         lag1_scaled_value = (lag1_unscaled - traffic_mean) / traffic_std
#         lag24_scaled_value = (lag24_unscaled - traffic_mean) / traffic_std
        
#         X_scaled_all = scaler.transform(X_numeric.values)

#         X_scaled_all[0, lag1_idx] = lag1_scaled_value
#         X_scaled_all[0, lag24_idx] = lag24_scaled_value
        
#         X_pred_row[NUMERIC_COLS_TO_SCALE] = X_scaled_all

#         # Predict and update cascade
#         dmatrix_X = xgb.DMatrix(X_pred_row) 
#         pred = model.predict(dmatrix_X)[0] 
#         pred = max(0, int(pred))
         
#         predicted_volumes.append(pred) 
#         previous_predicted_volume = pred
        
#     df_weather_data["Predicted_Traffic_Volume"] = predicted_volumes
#     return df_weather_data

# # --------------------------
# # App Title and Layout
# # --------------------------
# st.set_page_config(page_title="Traffic Prediction", page_icon="🚦", layout="wide")

# st.title(f"🚦 Traffic Volume Prediction for {FIXED_CITY_NAME}")
# st.markdown("---")

# # --------------------------
# # Sidebar and Checks
# # --------------------------
# if model and scaler:
#     MODEL_LOADED = True
# else:
#     MODEL_LOADED = False

# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.markdown("---")
    
#     if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
#         st.success("✅ API Key Loaded.")
#     else:
#         st.error("🔑 API Key not configured in secrets.")

#     if MODEL_LOADED:
#         st.success("✅ Model Assets Loaded.")
    
#     st.info(f"📍 **Location:** {FIXED_CITY_NAME}, {FIXED_COUNTRY}\nLat: {FIXED_LAT}, Lon: {FIXED_LON}")
    
#     # Fixed the TypeError by checking if traffic_mean is not None
#     if traffic_mean is not None:
#         st.markdown(f"📊 **Training Mean Traffic:** {traffic_mean:.0f} vehicles")

# # --------------------------
# # Prediction Button Logic (Corrected API Call Scope)
# # --------------------------
# if st.button("🔍 Predict Tomorrow's Traffic", type="primary"):
#     if not MODEL_LOADED or not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
#         st.error("Cannot proceed: Model/Scaler assets or API Key are missing.")
#     else:
#         with st.spinner("Fetching weather data and running model..."):
            
#             # --- 1. Define URL and API Fetch (Safely inside the button logic) ---
#             URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={FIXED_LAT}&lon={FIXED_LON}&appid={API_KEY}&units=metric"
            
#             try:
#                 response = requests.get(URL, timeout=10)
#                 response.raise_for_status()
#                 data = response.json()
#             except requests.exceptions.RequestException as e:
#                 st.error(f"❌ Failed to fetch weather data: {e}")
#                 st.info("Check your API key status and internet connection.")
#                 st.stop()
            
#             # --- 2. Data Extraction ---
#             forecast_list = []
#             today = datetime.datetime.now().date()
#             tomorrow = today + datetime.timedelta(days=1)
            
#             for entry in data["list"]:
#                 date_time = datetime.datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
#                 if date_time.date() == tomorrow:
#                     forecast_list.append({
#                         "datetime": date_time,
#                         "temp": entry["main"]["temp"],
#                         "feels_like": entry["main"]["feels_like"],
#                         "humidity": entry["main"]["humidity"],
#                         "pressure": entry["main"]["pressure"],
#                         "wind_speed": entry["wind"]["speed"],
#                         "clouds": entry["clouds"]["all"],
#                         "weather_main": entry["weather"][0]["main"],
#                         "weather_description": entry["weather"][0]["description"],
#                         "rain_1h": entry.get("rain", {}).get("3h", 0) / 3 
#                     })

#             if not forecast_list:
#                 st.warning("⚠️ No 3-hourly forecast data available for tomorrow.")
#                 st.stop()

#             # --- 3. Run Core Logic ---
#             df_weather_data = pd.DataFrame(forecast_list)
#             df_final = run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx)

#             # --------------------------
#             # Display Results
#             # --------------------------
            
#             st.success(f"✅ Forecast generated for **{tomorrow.strftime('%A, %B %d')}**")
            
#             # Prepare Display Data
#             display_df = df_final[["datetime", "weather_main", "temp", "Predicted_Traffic_Volume"]].copy()
#             display_df.columns = ["Date & Time", "Weather", "Temp (°C)", "Predicted Traffic Volume"]
#             display_df["Date & Time"] = display_df["Date & Time"].dt.strftime("%I:%M %p")
            
#             # Statistics
#             predictions = df_final["Predicted_Traffic_Volume"]
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("🚗 Average Traffic", f"{predictions.mean():,.0f} veh.")
#             with col2:
#                 st.metric("📈 Peak Traffic", f"{predictions.max():,.0f} veh.")
#             with col3:
#                 peak_index = predictions.argmax()
#                 peak_hour = display_df.loc[peak_index, "Date & Time"]
#                 st.metric("⏰ Peak Hour", peak_hour)
#             with col4:
#                 st.metric("Total Daily Volume", f"{predictions.sum():,.0f} veh.")

#             st.markdown("---")
#             st.subheader("Hourly Prediction Table")
#             st.dataframe(display_df, use_container_width=True)

#             # Visualization
#             st.subheader("📊 Traffic Volume Chart")
#             chart_data = df_final.copy()
#             chart_data["Time"] = chart_data["datetime"].dt.strftime("%I:%M %p")
#             chart_data = chart_data.set_index("Time")
#             st.line_chart(chart_data["Predicted_Traffic_Volume"], height=400)
            
#             # Download
#             csv = df_final.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Predictions as CSV",
#                 data=csv,
#                 file_name=f"traffic_predictions_{tomorrow}.csv",
#                 mime="text/csv"
#             )

# # --------------------------
# # Footer
# # --------------------------
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center'>
#         <p>Model based on Historical Ljubljana Traffic Data | Powered by OpenWeather API 🌤️</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )







# import streamlit as st
# import pickle
# import pandas as pd
# import requests
# import xgboost as xgb
# import datetime
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# # Suppress warnings related to setting values on DataFrame copies or scikit-learn feature names
# import warnings
# warnings.filterwarnings("ignore")

# # --------------------------
# # Global Configuration & Feature Lists
# # --------------------------
# FIXED_LAT = 46.0569
# FIXED_LON = 14.5058
# FIXED_CITY_NAME = "Ljubljana"
# FIXED_COUNTRY = "SI"
# # Use st.secrets to safely load the key
# API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE") 

# # --- Feature Lists (MUST match training structure) ---
# TRAINING_FEATURES = [
#     'temp', 'rain_1h', 'clouds_all', 'day_of_week', 'hour', 'month',
#     'traffic_volume_lag1', 'traffic_volume_lag24',
#     'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze',
#     'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
#     'weather_main_Squall', 'weather_main_Thunderstorm', 'weather_description_Sky is Clear', 
#     'weather_description_broken clouds', 'weather_description_drizzle', 'weather_description_few clouds', 
#     'weather_description_fog', 'weather_description_freezing rain', 'weather_description_haze', 
#     'weather_description_heavy intensity drizzle', 'weather_description_heavy intensity rain', 
#     'weather_description_heavy snow', 'weather_description_light intensity drizzle', 
#     'weather_description_light intensity shower rain', 'weather_description_light rain', 
#     'weather_description_light rain and snow', 'weather_description_light shower snow', 
#     'weather_description_light snow', 'weather_description_mist', 'weather_description_moderate rain', 
#     'weather_description_overcast clouds', 'weather_description_proximity shower rain', 
#     'weather_description_proximity thunderstorm', 'weather_description_proximity thunderstorm with drizzle',
#     'weather_description_proximity thunderstorm with rain', 'weather_description_scattered clouds', 
#     'weather_description_shower drizzle', 'weather_description_shower snow', 'weather_description_sky is clear', 
#     'weather_description_sleet', 'weather_description_smoke', 'weather_description_snow', 
#     'weather_description_thunderstorm', 'weather_description_thunderstorm with drizzle', 
#     'weather_description_thunderstorm with heavy rain', 'weather_description_thunderstorm with light drizzle',
#     'weather_description_thunderstorm with light rain', 'weather_description_thunderstorm with rain',
#     'weather_description_very heavy rain', 'is_holiday_True'
# ]

# NUMERIC_COLS_TO_SCALE = [
#     'temp', 'rain_1h', 'clouds_all', 'hour', 'day_of_week', 'month',
#     'traffic_volume_lag1', 'traffic_volume_lag24'
# ]

# HOURLY_MULTIPLIERS = {
#     0: 0.37, 1: 0.25, 2: 0.18, 3: 0.15, 4: 0.20, 5: 0.47,
#     6: 1.08, 7: 1.58, 8: 1.77, 9: 1.47, 10: 1.27, 11: 1.22,
#     12: 1.27, 13: 1.37, 14: 1.47, 15: 1.67, 16: 1.88, 17: 1.97,
#     18: 1.77, 19: 1.37, 20: 0.97, 21: 0.77, 22: 0.62, 23: 0.47
# }

# # --------------------------
# # Load Trained Model and Scaler
# # --------------------------

# @st.cache_resource
# def load_assets():
#     """Loads the model and scaler and extracts necessary stats."""
#     try:
#         model = xgb.Booster()
#         model.load_model("xgb_traffic_modell.json")
        
#         with open("scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
        
#         lag1_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag1')
#         traffic_mean = scaler.mean_[lag1_idx]
#         traffic_std = scaler.scale_[lag1_idx]
#         lag24_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag24')
        
#         return model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx
#     except xgb.core.XGBoostError as e:
#         st.error(f"❌ Could not load XGBoost model: {e}")
#         st.info("💡 Ensure 'xgb_traffic_modell.json' exists and is a valid XGBoost model file.")
#         return None, None, None, None, None, None
#     except Exception as e:
#         return None, None, None, None, None, None

# model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx = load_assets()

# # --------------------------
# # Main Prediction Function
# # --------------------------

# def get_hourly_baseline(traffic_mean):
#     """Calculates the hourly baseline traffic volumes."""
#     return {hour: traffic_mean * HOURLY_MULTIPLIERS.get(hour, 1.0) for hour in range(24)}

# def run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx):
    
#     df_features = df_weather_data.copy()
    
#     # 1. Feature Engineering
#     df_features["hour"] = df_features["datetime"].dt.hour
#     df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
#     df_features["month"] = df_features["datetime"].dt.month
#     df_features["clouds_all"] = df_features["clouds"]
#     df_features["is_holiday_True"] = 0
#     df_features['weather'] = df_features['weather_main']
#     df_features['weather_desc'] = df_features['weather_description'] 
    
#     df_encoded = pd.get_dummies(df_features, columns=["weather", "weather_desc"],  
#                                 prefix=["weather_main", "weather_description"], dtype=int) 
    
#     baseline_traffic = get_hourly_baseline(traffic_mean)
    
#     # 2. Prediction Loop (Pure Cascading)
#     predicted_volumes = [] 
#     previous_predicted_volume = baseline_traffic[23]

#     for i in range(len(df_encoded)): 
#         row = df_encoded.iloc[[i]].copy()
#         current_hour = row['hour'].values[0] 
        
#         # Lag Feature Assignment (Pure Cascading)
#         row["traffic_volume_lag1"] = previous_predicted_volume
#         row["traffic_volume_lag24"] = baseline_traffic[current_hour] 
        
#         # Align Features
#         X_pred_row = pd.DataFrame(0, index=[0], columns=TRAINING_FEATURES) 
#         for col in X_pred_row.columns: 
#             if col in row.columns: 
#                 X_pred_row[col] = row[col].values[0] 
        
#         # --- SCALING FIX: The critical 8-feature scaling ---
#         X_numeric = X_pred_row[NUMERIC_COLS_TO_SCALE].copy()
#         lag1_unscaled = X_numeric['traffic_volume_lag1'].values[0]
#         lag24_unscaled = X_numeric['traffic_volume_lag24'].values[0]
        
#         lag1_scaled_value = (lag1_unscaled - traffic_mean) / traffic_std
#         lag24_scaled_value = (lag24_unscaled - traffic_mean) / traffic_std
        
#         X_scaled_all = scaler.transform(X_numeric.values)

#         X_scaled_all[0, lag1_idx] = lag1_scaled_value
#         X_scaled_all[0, lag24_idx] = lag24_scaled_value
        
#         X_pred_row[NUMERIC_COLS_TO_SCALE] = X_scaled_all

#         # Predict and update cascade
#         dmatrix_X = xgb.DMatrix(X_pred_row) 
#         pred = model.predict(dmatrix_X)[0] 
#         pred = max(0, int(pred))
         
#         predicted_volumes.append(pred) 
#         previous_predicted_volume = pred
        
#     df_weather_data["Predicted_Traffic_Volume"] = predicted_volumes
#     return df_weather_data

# # --------------------------
# # App Title and Layout
# # --------------------------
# st.set_page_config(page_title="5-Day Traffic Prediction", page_icon="🚦", layout="wide")

# st.title(f"🚦 5-Day Traffic Volume Prediction for {FIXED_CITY_NAME}")
# st.markdown("---")

# # --------------------------
# # Sidebar and Checks
# # --------------------------
# if model and scaler:
#     MODEL_LOADED = True
# else:
#     MODEL_LOADED = False

# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.markdown("---")
    
#     if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
#         st.success("✅ API Key Loaded.")
#     else:
#         st.error("🔑 API Key not configured in secrets.")

#     if MODEL_LOADED:
#         st.success("✅ Model Assets Loaded.")
    
#     st.info(f"📍 **Location:** {FIXED_CITY_NAME}, {FIXED_COUNTRY}\nLat: {FIXED_LAT}, Lon: {FIXED_LON}")
    
#     if traffic_mean is not None:
#         st.markdown(f"📊 **Training Mean Traffic:** {traffic_mean:.0f} vehicles")

# # --------------------------
# # Prediction Button Logic
# # --------------------------
# if st.button("🔍 Predict Next 5 Days Traffic", type="primary"):
#     if not MODEL_LOADED or not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
#         st.error("Cannot proceed: Model/Scaler assets or API Key are missing.")
#     else:
#         with st.spinner("Fetching weather data and running model..."):
            
#             # --- 1. Define URL and API Fetch ---
#             URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={FIXED_LAT}&lon={FIXED_LON}&appid={API_KEY}&units=metric"
            
#             try:
#                 response = requests.get(URL, timeout=10)
#                 response.raise_for_status()
#                 data = response.json()
#             except requests.exceptions.RequestException as e:
#                 st.error(f"❌ Failed to fetch weather data: {e}")
#                 st.info("Check your API key status and internet connection.")
#                 st.stop()
            
#             # --- 2. Data Extraction (5 days) ---
#             forecast_list = []
#             today = datetime.datetime.now().date()
            
#             for entry in data["list"]:
#                 date_time = datetime.datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
#                 if date_time.date() > today:  # Get all future data
#                     forecast_list.append({
#                         "datetime": date_time,
#                         "temp": entry["main"]["temp"],
#                         "feels_like": entry["main"]["feels_like"],
#                         "humidity": entry["main"]["humidity"],
#                         "pressure": entry["main"]["pressure"],
#                         "wind_speed": entry["wind"]["speed"],
#                         "clouds": entry["clouds"]["all"],
#                         "weather_main": entry["weather"][0]["main"],
#                         "weather_description": entry["weather"][0]["description"],
#                         "rain_1h": entry.get("rain", {}).get("3h", 0) / 3 
#                     })

#             if not forecast_list:
#                 st.warning("⚠️ No forecast data available.")
#                 st.stop()

#             # --- 3. Run Core Logic ---
#             df_weather_data = pd.DataFrame(forecast_list)
#             df_final = run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx)

#             # Add date column for grouping
#             df_final['date'] = df_final['datetime'].dt.date
#             df_final['day_name'] = df_final['datetime'].dt.strftime('%A, %B %d')
            
#             # --------------------------
#             # Calculate Daily Statistics
#             # --------------------------
#             daily_stats = df_final.groupby('date').agg({
#                 'Predicted_Traffic_Volume': ['sum', 'mean', 'max'],
#                 'day_name': 'first'
#             }).reset_index()
            
#             daily_stats.columns = ['date', 'total_volume', 'avg_volume', 'peak_volume', 'day_name']
#             daily_stats = daily_stats.sort_values('date')
            
#             # Find highest traffic day
#             highest_traffic_day = daily_stats.loc[daily_stats['total_volume'].idxmax()]
            
#             # --------------------------
#             # Display Overall Summary
#             # --------------------------
#             st.success(f"✅ 5-Day Forecast Generated ({daily_stats['date'].min()} to {daily_stats['date'].max()})")
            
#             st.markdown("### 📊 5-Day Overview")
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 total_5day = daily_stats['total_volume'].sum()
#                 st.metric("🚗 Total 5-Day Volume", f"{total_5day:,.0f} veh.")
#             with col2:
#                 avg_daily = daily_stats['total_volume'].mean()
#                 st.metric("📈 Avg Daily Volume", f"{avg_daily:,.0f} veh.")
#             with col3:
#                 st.metric("🔥 Highest Traffic Day", highest_traffic_day['day_name'].split(',')[0])
#             with col4:
#                 st.metric("🔥 Highest Day Volume", f"{highest_traffic_day['total_volume']:,.0f} veh.")
            
#             st.markdown("---")
            
#             # --------------------------
#             # Daily Breakdown Cards
#             # --------------------------
#             st.markdown("### 📅 Daily Traffic Summary")
            
#             for idx, day_row in daily_stats.iterrows():
#                 is_highest = day_row['date'] == highest_traffic_day['date']
                
#                 with st.expander(
#                     f"{'🔥 ' if is_highest else '📅 '}{day_row['day_name']} - Total: {day_row['total_volume']:,.0f} vehicles",
#                     expanded=is_highest
#                 ):
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Total Daily Volume", f"{day_row['total_volume']:,.0f} veh.")
#                     with col2:
#                         st.metric("Average Hourly", f"{day_row['avg_volume']:,.0f} veh.")
#                     with col3:
#                         st.metric("Peak Hour Volume", f"{day_row['peak_volume']:,.0f} veh.")
                    
#                     # Filter data for this day
#                     day_data = df_final[df_final['date'] == day_row['date']].copy()
#                     day_data['time'] = day_data['datetime'].dt.strftime('%H:%M')
                    
#                     # Create line chart for this day
#                     fig = px.line(
#                         day_data, 
#                         x='time', 
#                         y='Predicted_Traffic_Volume',
#                         title=f"Hourly Traffic Pattern - {day_row['day_name']}",
#                         labels={'time': 'Time', 'Predicted_Traffic_Volume': 'Traffic Volume'},
#                         markers=True
#                     )
#                     fig.update_layout(height=300, showlegend=False)
#                     st.plotly_chart(fig, use_container_width=True)
            
#             st.markdown("---")
            
#             # --------------------------
#             # Overall 5-Day Chart
#             # --------------------------
#             st.markdown("### 📈 Complete 5-Day Traffic Pattern")
            
#             df_chart = df_final.copy()
#             df_chart['datetime_str'] = df_chart['datetime'].dt.strftime('%m/%d %H:%M')
            
#             fig_overall = go.Figure()
            
#             # Add line for each day with different colors
#             for date in daily_stats['date']:
#                 day_data = df_chart[df_chart['date'] == date]
#                 day_name = day_data['day_name'].iloc[0].split(',')[0]
                
#                 fig_overall.add_trace(go.Scatter(
#                     x=day_data['datetime_str'],
#                     y=day_data['Predicted_Traffic_Volume'],
#                     mode='lines+markers',
#                     name=day_name,
#                     hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Traffic: %{y:,.0f} vehicles<extra></extra>'
#                 ))
            
#             fig_overall.update_layout(
#                 title="5-Day Traffic Volume Forecast",
#                 xaxis_title="Date & Time",
#                 yaxis_title="Traffic Volume (vehicles)",
#                 height=500,
#                 hovermode='x unified'
#             )
            
#             st.plotly_chart(fig_overall, use_container_width=True)
            
#             # --------------------------
#             # Daily Comparison Bar Chart
#             # --------------------------
#             st.markdown("### 📊 Daily Volume Comparison")
            
#             fig_bar = px.bar(
#                 daily_stats,
#                 x='day_name',
#                 y='total_volume',
#                 title="Total Daily Traffic Volume",
#                 labels={'day_name': 'Day', 'total_volume': 'Total Volume'},
#                 color='total_volume',
#                 color_continuous_scale='Reds'
#             )
#             fig_bar.update_layout(height=400, showlegend=False)
#             st.plotly_chart(fig_bar, use_container_width=True)
            
#             st.markdown("---")
            
#             # --------------------------
#             # Detailed Hourly Table
#             # --------------------------
#             st.markdown("### 📋 Detailed Hourly Predictions")
            
#             display_df = df_final[["datetime", "day_name", "weather_main", "temp", "Predicted_Traffic_Volume"]].copy()
#             display_df.columns = ["Date & Time", "Day", "Weather", "Temp (°C)", "Traffic Volume"]
#             display_df["Date & Time"] = display_df["Date & Time"].dt.strftime("%m/%d %I:%M %p")
            
#             st.dataframe(display_df, use_container_width=True, height=400)
            
#             # --------------------------
#             # Download Options
#             # --------------------------
#             st.markdown("### 💾 Download Data")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 csv_full = df_final.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Full Predictions (CSV)",
#                     data=csv_full,
#                     file_name=f"traffic_predictions_5day_{today}.csv",
#                     mime="text/csv"
#                 )
            
#             with col2:
#                 csv_summary = daily_stats.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Daily Summary (CSV)",
#                     data=csv_summary,
#                     file_name=f"traffic_daily_summary_{today}.csv",
#                     mime="text/csv"
#                 )

# # --------------------------
# # Footer
# # --------------------------
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center'>
#         <p>Model based on Historical Ljubljana Traffic Data | Powered by OpenWeather API 🌤️</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )




import streamlit as st
import pickle
import pandas as pd
import requests
import xgboost as xgb
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings related to setting values on DataFrame copies or scikit-learn feature names
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Global Configuration & Feature Lists
# --------------------------
FIXED_LAT = 46.0569
FIXED_LON = 14.5058
FIXED_CITY_NAME = "Ljubljana"
FIXED_COUNTRY = "SI"
# Use st.secrets to safely load the key
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE") 

# --- Feature Lists (MUST match training structure) ---
TRAINING_FEATURES = [
    'temp', 'rain_1h', 'clouds_all', 'day_of_week', 'hour', 'month',
    'traffic_volume_lag1', 'traffic_volume_lag24',
    'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze',
    'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
    'weather_main_Squall', 'weather_main_Thunderstorm', 'weather_description_Sky is Clear', 
    'weather_description_broken clouds', 'weather_description_drizzle', 'weather_description_few clouds', 
    'weather_description_fog', 'weather_description_freezing rain', 'weather_description_haze', 
    'weather_description_heavy intensity drizzle', 'weather_description_heavy intensity rain', 
    'weather_description_heavy snow', 'weather_description_light intensity drizzle', 
    'weather_description_light intensity shower rain', 'weather_description_light rain', 
    'weather_description_light rain and snow', 'weather_description_light shower snow', 
    'weather_description_light snow', 'weather_description_mist', 'weather_description_moderate rain', 
    'weather_description_overcast clouds', 'weather_description_proximity shower rain', 
    'weather_description_proximity thunderstorm', 'weather_description_proximity thunderstorm with drizzle',
    'weather_description_proximity thunderstorm with rain', 'weather_description_scattered clouds', 
    'weather_description_shower drizzle', 'weather_description_shower snow', 'weather_description_sky is clear', 
    'weather_description_sleet', 'weather_description_smoke', 'weather_description_snow', 
    'weather_description_thunderstorm', 'weather_description_thunderstorm with drizzle', 
    'weather_description_thunderstorm with heavy rain', 'weather_description_thunderstorm with light drizzle',
    'weather_description_thunderstorm with light rain', 'weather_description_thunderstorm with rain',
    'weather_description_very heavy rain', 'is_holiday_True'
]

NUMERIC_COLS_TO_SCALE = [
    'temp', 'rain_1h', 'clouds_all', 'hour', 'day_of_week', 'month',
    'traffic_volume_lag1', 'traffic_volume_lag24'
]

HOURLY_MULTIPLIERS = {
    0: 0.37, 1: 0.25, 2: 0.18, 3: 0.15, 4: 0.20, 5: 0.47,
    6: 1.08, 7: 1.58, 8: 1.77, 9: 1.47, 10: 1.27, 11: 1.22,
    12: 1.27, 13: 1.37, 14: 1.47, 15: 1.67, 16: 1.88, 17: 1.97,
    18: 1.77, 19: 1.37, 20: 0.97, 21: 0.77, 22: 0.62, 23: 0.47
}

# --------------------------
# Load Trained Model and Scaler
# --------------------------

@st.cache_resource
def load_assets():
    """Loads the model and scaler and extracts necessary stats."""
    try:
        model = xgb.Booster()
        model.load_model("xgb_traffic_modell.json")
        
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        lag1_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag1')
        traffic_mean = scaler.mean_[lag1_idx]
        traffic_std = scaler.scale_[lag1_idx]
        lag24_idx = NUMERIC_COLS_TO_SCALE.index('traffic_volume_lag24')
        
        return model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx
    except xgb.core.XGBoostError as e:
        st.error(f"❌ Could not load XGBoost model: {e}")
        st.info("💡 Ensure 'xgb_traffic_modell.json' exists and is a valid XGBoost model file.")
        return None, None, None, None, None, None
    except Exception as e:
        return None, None, None, None, None, None

model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx = load_assets()

# --------------------------
# Main Prediction Function
# --------------------------

def get_hourly_baseline(traffic_mean):
    """Calculates the hourly baseline traffic volumes."""
    return {hour: traffic_mean * HOURLY_MULTIPLIERS.get(hour, 1.0) for hour in range(24)}

def run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx):
    
    df_features = df_weather_data.copy()
    
    # 1. Feature Engineering
    df_features["hour"] = df_features["datetime"].dt.hour
    df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
    df_features["month"] = df_features["datetime"].dt.month
    df_features["clouds_all"] = df_features["clouds"]
    df_features["is_holiday_True"] = 0
    df_features['weather'] = df_features['weather_main']
    df_features['weather_desc'] = df_features['weather_description'] 
    
    df_encoded = pd.get_dummies(df_features, columns=["weather", "weather_desc"],  
                                prefix=["weather_main", "weather_description"], dtype=int) 
    
    baseline_traffic = get_hourly_baseline(traffic_mean)
    
    # 2. Prediction Loop (Pure Cascading)
    predicted_volumes = [] 
    previous_predicted_volume = baseline_traffic[23]

    for i in range(len(df_encoded)): 
        row = df_encoded.iloc[[i]].copy()
        current_hour = row['hour'].values[0] 
        
        # Lag Feature Assignment (Pure Cascading)
        row["traffic_volume_lag1"] = previous_predicted_volume
        row["traffic_volume_lag24"] = baseline_traffic[current_hour] 
        
        # Align Features
        X_pred_row = pd.DataFrame(0, index=[0], columns=TRAINING_FEATURES) 
        for col in X_pred_row.columns: 
            if col in row.columns: 
                X_pred_row[col] = row[col].values[0] 
        
        # --- SCALING FIX: The critical 8-feature scaling ---
        X_numeric = X_pred_row[NUMERIC_COLS_TO_SCALE].copy()
        lag1_unscaled = X_numeric['traffic_volume_lag1'].values[0]
        lag24_unscaled = X_numeric['traffic_volume_lag24'].values[0]
        
        lag1_scaled_value = (lag1_unscaled - traffic_mean) / traffic_std
        lag24_scaled_value = (lag24_unscaled - traffic_mean) / traffic_std
        
        X_scaled_all = scaler.transform(X_numeric.values)

        X_scaled_all[0, lag1_idx] = lag1_scaled_value
        X_scaled_all[0, lag24_idx] = lag24_scaled_value
        
        X_pred_row[NUMERIC_COLS_TO_SCALE] = X_scaled_all

        # Predict and update cascade
        dmatrix_X = xgb.DMatrix(X_pred_row) 
        pred = model.predict(dmatrix_X)[0] 
        pred = max(0, int(pred))
         
        predicted_volumes.append(pred) 
        previous_predicted_volume = pred
        
    df_weather_data["Predicted_Traffic_Volume"] = predicted_volumes
    return df_weather_data

# --------------------------
# App Title and Layout
# --------------------------
st.set_page_config(page_title="5-Day Traffic Prediction", page_icon="🚦", layout="wide")

st.title(f"🚦 5-Day Traffic Volume Prediction for {FIXED_CITY_NAME}")
st.markdown("---")

# --------------------------
# Sidebar and Checks
# --------------------------
if model and scaler:
    MODEL_LOADED = True
else:
    MODEL_LOADED = False

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
        st.success("✅ API Key Loaded.")
    else:
        st.error("🔑 API Key not configured in secrets.")

    if MODEL_LOADED:
        st.success("✅ Model Assets Loaded.")
    
    st.info(f"📍 **Location:** {FIXED_CITY_NAME}, {FIXED_COUNTRY}\nLat: {FIXED_LAT}, Lon: {FIXED_LON}")
    
    if traffic_mean is not None:
        st.markdown(f"📊 **Training Mean Traffic:** {traffic_mean:.0f} vehicles")

# --------------------------
# Prediction Button Logic
# --------------------------
if st.button("🔍 Predict Next 5 Days Traffic", type="primary"):
    if not MODEL_LOADED or not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        st.error("Cannot proceed: Model/Scaler assets or API Key are missing.")
    else:
        with st.spinner("Fetching weather data and running model..."):
            
            # --- 1. Define URL and API Fetch ---
            URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={FIXED_LAT}&lon={FIXED_LON}&appid={API_KEY}&units=metric"
            
            try:
                response = requests.get(URL, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to fetch weather data: {e}")
                st.info("Check your API key status and internet connection.")
                st.stop()
            
            # --- 2. Data Extraction (5 days) ---
            forecast_list = []
            today = datetime.datetime.now().date()
            
            for entry in data["list"]:
                date_time = datetime.datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
                if date_time.date() > today:  # Get all future data
                    forecast_list.append({
                        "datetime": date_time,
                        "temp": entry["main"]["temp"],
                        "feels_like": entry["main"]["feels_like"],
                        "humidity": entry["main"]["humidity"],
                        "pressure": entry["main"]["pressure"],
                        "wind_speed": entry["wind"]["speed"],
                        "clouds": entry["clouds"]["all"],
                        "weather_main": entry["weather"][0]["main"],
                        "weather_description": entry["weather"][0]["description"],
                        "rain_1h": entry.get("rain", {}).get("3h", 0) / 3 
                    })

            if not forecast_list:
                st.warning("⚠️ No forecast data available.")
                st.stop()

            # --- 3. Run Core Logic ---
            df_weather_data = pd.DataFrame(forecast_list)
            df_final = run_prediction_loop(df_weather_data, model, scaler, traffic_mean, traffic_std, lag1_idx, lag24_idx)

            # Add date column for grouping
            df_final['date'] = df_final['datetime'].dt.date
            df_final['day_name'] = df_final['datetime'].dt.strftime('%A, %B %d')
            
            # --------------------------
            # Calculate Daily Statistics
            # --------------------------
            daily_stats = df_final.groupby('date').agg({
                'Predicted_Traffic_Volume': ['sum', 'mean', 'max'],
                'day_name': 'first'
            }).reset_index()
            
            daily_stats.columns = ['date', 'total_volume', 'avg_volume', 'peak_volume', 'day_name']
            daily_stats = daily_stats.sort_values('date')
            
            # Find highest traffic day
            highest_traffic_day = daily_stats.loc[daily_stats['total_volume'].idxmax()]
            
            # --------------------------
            # Display Overall Summary
            # --------------------------
            st.success(f"✅ 5-Day Forecast Generated ({daily_stats['date'].min()} to {daily_stats['date'].max()})")
            
            st.markdown("### 📊 5-Day Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_5day = daily_stats['total_volume'].sum()
                st.metric("🚗 Total 5-Day Volume", f"{total_5day:,.0f} veh.")
            with col2:
                avg_daily = daily_stats['total_volume'].mean()
                st.metric("📈 Avg Daily Volume", f"{avg_daily:,.0f} veh.")
            with col3:
                st.metric("🔥 Highest Traffic Day", highest_traffic_day['day_name'].split(',')[0])
            with col4:
                st.metric("🔥 Highest Day Volume", f"{highest_traffic_day['total_volume']:,.0f} veh.")
            
            st.markdown("---")
            
            # --------------------------
            # Daily Breakdown Cards
            # --------------------------
            st.markdown("### 📅 Daily Traffic Summary")
            
            for idx, day_row in daily_stats.iterrows():
                is_highest = day_row['date'] == highest_traffic_day['date']
                
                with st.expander(
                    f"{'🔥 ' if is_highest else '📅 '}{day_row['day_name']} - Total: {day_row['total_volume']:,.0f} vehicles",
                    expanded=is_highest
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Daily Volume", f"{day_row['total_volume']:,.0f} veh.")
                    with col2:
                        st.metric("Average Hourly", f"{day_row['avg_volume']:,.0f} veh.")
                    with col3:
                        st.metric("Peak Hour Volume", f"{day_row['peak_volume']:,.0f} veh.")
                    
                    # Filter data for this day
                    day_data = df_final[df_final['date'] == day_row['date']].copy()
                    day_data['time'] = day_data['datetime'].dt.strftime('%H:%M')
                    
                    # Create line chart for this day
                    fig = px.line(
                        day_data, 
                        x='time', 
                        y='Predicted_Traffic_Volume',
                        title=f"Hourly Traffic Pattern - {day_row['day_name']}",
                        labels={'time': 'Time', 'Predicted_Traffic_Volume': 'Traffic Volume'},
                        markers=True
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # --------------------------
            # Overall 5-Day Chart - Improved Version
            # --------------------------
            st.markdown("### 📈 Complete 5-Day Traffic Pattern")
            
            # Create a normalized time column (hour only) for better x-axis
            df_chart = df_final.copy()
            df_chart['hour_only'] = df_chart['datetime'].dt.hour
            df_chart['time_label'] = df_chart['datetime'].dt.strftime('%H:%M')
            
            fig_overall = go.Figure()
            
            # Add line for each day with different colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for idx, date in enumerate(daily_stats['date']):
                day_data = df_chart[df_chart['date'] == date].copy()
                day_name = day_data['day_name'].iloc[0].split(',')[0]
                
                # Use hour as x-axis for cleaner display
                fig_overall.add_trace(go.Scatter(
                    x=day_data['hour_only'],
                    y=day_data['Predicted_Traffic_Volume'],
                    mode='lines+markers',
                    name=day_name,
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    marker=dict(size=4),
                    hovertemplate='<b>%{fullData.name}</b><br>Hour: %{x}:00<br>Traffic: %{y:,.0f} vehicles<extra></extra>'
                ))
            
            fig_overall.update_layout(
                title="5-Day Traffic Volume Forecast (Hourly Pattern)",
                xaxis_title="Hour of Day",
                yaxis_title="Traffic Volume (vehicles)",
                height=500,
                hovermode='closest',
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=3,
                    tickformat='%H:00'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_overall, use_container_width=True)
            
            # --------------------------
            # Daily Comparison Bar Chart
            # --------------------------
            st.markdown("### 📊 Daily Volume Comparison")
            
            fig_bar = px.bar(
                daily_stats,
                x='day_name',
                y='total_volume',
                title="Total Daily Traffic Volume",
                labels={'day_name': 'Day', 'total_volume': 'Total Volume'},
                color='total_volume',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # --------------------------
            # Detailed Hourly Table
            # --------------------------
            st.markdown("### 📋 Detailed Hourly Predictions")
            
            display_df = df_final[["datetime", "day_name", "weather_main", "temp", "Predicted_Traffic_Volume"]].copy()
            display_df.columns = ["Date & Time", "Day", "Weather", "Temp (°C)", "Traffic Volume"]
            display_df["Date & Time"] = display_df["Date & Time"].dt.strftime("%m/%d %I:%M %p")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # --------------------------
            # Download Options
            # --------------------------
            st.markdown("### 💾 Download Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_full = df_final.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Predictions (CSV)",
                    data=csv_full,
                    file_name=f"traffic_predictions_5day_{today}.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_summary = daily_stats.to_csv(index=False)
                st.download_button(
                    label="📥 Download Daily Summary (CSV)",
                    data=csv_summary,
                    file_name=f"traffic_daily_summary_{today}.csv",
                    mime="text/csv"
                )

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Model based on Historical Ljubljana Traffic Data | Powered by OpenWeather API 🌤️</p>
    </div>
    """,
    unsafe_allow_html=True
)