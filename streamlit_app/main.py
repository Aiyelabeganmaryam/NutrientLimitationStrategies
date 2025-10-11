import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Nutrient Limitation Strategies & Spatial Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Translations
translations = {
    'en': {
        'title': 'Nutrient Limitation Strategies & Spatial Analysis',
        'subtitle': 'Advanced Spatial Analysis for Agricultural Nutrient Management',
        'user_type': 'Select User Type',
        'farmer': 'Farmer',
        'researcher': 'Researcher/Scientist',
        'extension': 'Extension Agent',
        'policy': 'Policy Maker',
        'language': 'Language',
        'analysis': 'Analysis Dashboard',
        'data_input': 'Data Input',
        'results': 'Results & Insights',
        'about': 'About'
    },
    'ha': {
        'title': 'Dabarun Ƙarancin Abinci da Nazarin Yanki',
        'subtitle': 'Nazarin Yanki mai Zurfi don Sarrafa Abincin Noma',
        'user_type': 'Zaɓi Nau\'in Mai Amfani',
        'farmer': 'Manomi',
        'researcher': 'Mai Bincike/Masanin Kimiyya',
        'extension': 'Wakilin Koyarwa',
        'policy': 'Mai Yin Manufa',
        'language': 'Harshe',
        'analysis': 'Dashboard na Bincike',
        'data_input': 'Shigar da Bayanai',
        'results': 'Sakamako da Basirar',
        'about': 'Game da'
    }
}

def get_text(key):
    return translations[st.session_state.language].get(key, key)

@st.cache_data
def get_nigeria_lga_data():
    """Get sample Nigerian LGA data with real coordinates"""
    # Real Nigerian LGA coordinates (sample of major LGAs)
    real_lgas = [
        {'LGA': 'Kano Municipal', 'State': 'Kano', 'Latitude': 12.0022, 'Longitude': 8.5919},
        {'LGA': 'Fagge', 'State': 'Kano', 'Latitude': 12.0022, 'Longitude': 8.5180},
        {'LGA': 'Dala', 'State': 'Kano', 'Latitude': 11.9910, 'Longitude': 8.4950},
        {'LGA': 'Gwale', 'State': 'Kano', 'Latitude': 12.0130, 'Longitude': 8.5100},
        {'LGA': 'Nasarawa', 'State': 'Kano', 'Latitude': 12.0190, 'Longitude': 8.5320},
        {'LGA': 'Kaduna North', 'State': 'Kaduna', 'Latitude': 10.5105, 'Longitude': 7.4165},
        {'LGA': 'Kaduna South', 'State': 'Kaduna', 'Latitude': 10.5105, 'Longitude': 7.4264},
        {'LGA': 'Chikun', 'State': 'Kaduna', 'Latitude': 10.5922, 'Longitude': 7.3719},
        {'LGA': 'Igabi', 'State': 'Kaduna', 'Latitude': 10.8436, 'Longitude': 7.3719},
        {'LGA': 'Katsina', 'State': 'Katsina', 'Latitude': 13.0059, 'Longitude': 7.6006},
        {'LGA': 'Dutse', 'State': 'Jigawa', 'Latitude': 11.7564, 'Longitude': 9.3380},
        {'LGA': 'Bauchi', 'State': 'Bauchi', 'Latitude': 10.3158, 'Longitude': 9.8442},
        {'LGA': 'Sokoto North', 'State': 'Sokoto', 'Latitude': 13.0569, 'Longitude': 5.2476},
        {'LGA': 'Sokoto South', 'State': 'Sokoto', 'Latitude': 13.0569, 'Longitude': 5.2334},
        {'LGA': 'Gusau', 'State': 'Zamfara', 'Latitude': 12.1704, 'Longitude': 6.6593},
        {'LGA': 'Maiduguri', 'State': 'Borno', 'Latitude': 11.8311, 'Longitude': 13.1511},
        {'LGA': 'Jos North', 'State': 'Plateau', 'Latitude': 9.8965, 'Longitude': 8.8583},
        {'LGA': 'Zaria', 'State': 'Kaduna', 'Latitude': 11.0804, 'Longitude': 7.7076},
        {'LGA': 'Gombe', 'State': 'Gombe', 'Latitude': 10.2897, 'Longitude': 11.1689},
        {'LGA': 'Yola North', 'State': 'Adamawa', 'Latitude': 9.2094, 'Longitude': 12.4882}
    ]
    
    return pd.DataFrame(real_lgas)

@st.cache_data
def generate_sample_data():
    """Generate sample agricultural data for demonstration with real LGA coordinates"""
    np.random.seed(42)
    
    # Get real LGA data
    lga_data = get_nigeria_lga_data()
    n_samples = len(lga_data)
    
    # Add noise to coordinates for variation within LGAs
    lat_noise = np.random.normal(0, 0.05, n_samples)
    lon_noise = np.random.normal(0, 0.05, n_samples)
    
    data = {
        'LGA': lga_data['LGA'].values,
        'State': lga_data['State'].values,
        'Latitude': lga_data['Latitude'].values + lat_noise,
        'Longitude': lga_data['Longitude'].values + lon_noise,
        'Soil_pH': np.random.normal(6.2, 0.8, n_samples),
        'Organic_Matter_pct': np.random.lognormal(0.9, 0.4, n_samples),
        'N_ppm': np.random.normal(25, 8, n_samples),
        'P_ppm': np.random.normal(15, 6, n_samples),
        'K_ppm': np.random.normal(120, 25, n_samples),
        'Rainfall_mm': np.random.normal(750, 200, n_samples),
        'Temperature_C': np.random.normal(28, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create nutrient limitations based on thresholds
    df['N_Limited'] = (df['N_ppm'] < 20) | (df['Organic_Matter_pct'] < 1.5)
    df['P_Limited'] = (df['P_ppm'] < 10) | (df['Soil_pH'] < 5.5)
    df['K_Limited'] = (df['K_ppm'] < 80) | (df['Rainfall_mm'] > 1000)
    
    return df

def create_nigeria_map(data, nutrient_column, title):
    """Create an interactive map of Nigeria with nutrient data"""
    
    # Create base map centered on Nigeria
    center_lat = data['Latitude'].mean()
    center_lon = data['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for limitations
    def get_color(limited):
        return 'red' if limited else 'green'
    
    def get_popup_text(row, nutrient):
        limitation_status = "Limited" if row[nutrient_column] else "Adequate"
        color_indicator = "🔴" if row[nutrient_column] else "🟢"
        
        return f"""
        <b>{row['LGA']}, {row['State']}</b><br>
        {nutrient} Status: {color_indicator} {limitation_status}<br>
        pH: {row['Soil_pH']:.1f}<br>
        Organic Matter: {row['Organic_Matter_pct']:.1f}%<br>
        Rainfall: {row['Rainfall_mm']:.0f}mm<br>
        Temperature: {row['Temperature_C']:.1f}°C
        """
    
    # Add markers for each LGA
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=folium.Popup(
                get_popup_text(row, nutrient_column.replace('_Limited', '')),
                max_width=300
            ),
            color='darkred' if row[nutrient_column] else 'darkgreen',
            fillColor=get_color(row[nutrient_column]),
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{title}</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Adequate</p>
    <p><i class="fa fa-circle" style="color:red"></i> Limited</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_state_summary_map(data):
    """Create a choropleth-style map showing state-level nutrient summaries"""
    
    # Calculate state-level statistics
    state_stats = data.groupby('State').agg({
        'N_Limited': 'mean',
        'P_Limited': 'mean',
        'K_Limited': 'mean',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    # Create map
    center_lat = state_stats['Latitude'].mean()
    center_lon = state_stats['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add state markers with summary information
    for idx, row in state_stats.iterrows():
        # Calculate overall limitation risk
        avg_limitation = (row['N_Limited'] + row['P_Limited'] + row['K_Limited']) / 3
        
        # Color based on average limitation
        if avg_limitation > 0.6:
            color = 'red'
            risk_level = 'High Risk'
        elif avg_limitation > 0.3:
            color = 'orange' 
            risk_level = 'Medium Risk'
        else:
            color = 'green'
            risk_level = 'Low Risk'
        
        popup_text = f"""
        <b>{row['State']} State Summary</b><br>
        Overall Risk: <b>{risk_level}</b><br><br>
        <b>Limitation Frequencies:</b><br>
        🔴 Nitrogen: {row['N_Limited']:.1%}<br>
        🟡 Phosphorus: {row['P_Limited']:.1%}<br>
        🔵 Potassium: {row['K_Limited']:.1%}<br>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=15,
            popup=folium.Popup(popup_text, max_width=250),
            color='darkblue',
            fillColor=color,
            fillOpacity=0.7,
            weight=3
        ).add_to(m)
    
    return m
    
    return df

@st.cache_resource
def train_models(data):
    """Train machine learning models for nutrient prediction"""
    features = ['Soil_pH', 'Organic_Matter_pct', 'Rainfall_mm', 'Temperature_C']
    X = data[features]
    
    models = {}
    scores = {}
    
    for nutrient in ['N', 'P', 'K']:
        target = f'{nutrient}_Limited'
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        models[nutrient] = model
        scores[nutrient] = accuracy
    
    return models, scores, features

def main():
    # Sidebar
    with st.sidebar:
        st.title("🌱 " + get_text('title')[:25] + "...")
        
        # Language selector
        lang_options = {'English': 'en', 'Hausa': 'ha'}
        selected_lang = st.selectbox(
            get_text('language'),
            options=list(lang_options.keys()),
            index=0 if st.session_state.language == 'en' else 1
        )
        st.session_state.language = lang_options[selected_lang]
        
        # User type selector
        user_types = {
            get_text('farmer'): 'farmer',
            get_text('researcher'): 'researcher', 
            get_text('extension'): 'extension',
            get_text('policy'): 'policy'
        }
        
        selected_user = st.selectbox(
            get_text('user_type'),
            options=list(user_types.keys())
        )
        user_type = user_types[selected_user]
    
    # Main content
    st.title("🌱 " + get_text('title'))
    st.markdown(f"### {get_text('subtitle')}")
    
    # Load data and models
    with st.spinner('Loading data and training models...'):
        df = generate_sample_data()
        models, scores, features = train_models(df)
    
    # Create tabs
    if user_type == 'farmer':
        tabs = st.tabs([get_text('data_input'), get_text('results')])
        
        with tabs[0]:
            st.header("🌾 Farm Data Input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
                organic_matter = st.slider("Organic Matter (%)", 0.5, 5.0, 2.0, 0.1)
            
            with col2:
                rainfall = st.slider("Annual Rainfall (mm)", 300, 1500, 750, 10)
                temperature = st.slider("Average Temperature (°C)", 20, 35, 28, 1)
            
            if st.button("🔍 Analyze Nutrient Status", type="primary"):
                input_data = [ph, organic_matter, rainfall, temperature]
                
                st.subheader("🎯 Nutrient Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                for i, (nutrient, model) in enumerate(models.items()):
                    prediction = model.predict([input_data])[0]
                    probability = model.predict_proba([input_data])[0][1]
                    
                    status = "⚠️ Limited" if prediction else "✅ Adequate"
                    color = "red" if prediction else "green"
                    
                    with [col1, col2, col3][i]:
                        st.metric(
                            f"{nutrient} Status",
                            status,
                            f"Risk: {probability:.1%}",
                            delta_color="inverse"
                        )
                
                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = []
                
                for nutrient, model in models.items():
                    if model.predict([input_data])[0]:
                        if nutrient == 'N':
                            recommendations.append("🔴 **Nitrogen deficiency detected** - Apply Urea (100-150 kg/ha)")
                        elif nutrient == 'P':
                            recommendations.append("🟡 **Phosphorus deficiency detected** - Apply SSP (75-100 kg/ha)")
                        elif nutrient == 'K':
                            recommendations.append("🟠 **Potassium deficiency detected** - Apply Muriate of Potash (50-75 kg/ha)")
                
                if not recommendations:
                    st.success("✅ **No major nutrient deficiencies detected** - Continue with balanced fertilization")
                else:
                    for rec in recommendations:
                        st.markdown(rec)
        
        with tabs[1]:
            st.header("📊 Analysis Results")
            
            # Summary statistics
            st.subheader("🌍 Regional Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_limited = df['N_Limited'].mean() * 100
                st.metric("Nitrogen Limitation", f"{n_limited:.1f}%")
            
            with col2:
                p_limited = df['P_Limited'].mean() * 100
                st.metric("Phosphorus Limitation", f"{p_limited:.1f}%")
            
            with col3:
                k_limited = df['K_Limited'].mean() * 100
                st.metric("Potassium Limitation", f"{k_limited:.1f}%")
            
            # Visualizations
            fig1 = px.bar(
                x=['Nitrogen', 'Phosphorus', 'Potassium'],
                y=[df['N_Limited'].mean(), df['P_Limited'].mean(), df['K_Limited'].mean()],
                title="Nutrient Limitation Frequency",
                labels={'x': 'Nutrient', 'y': 'Limitation Frequency'},
                color=['Nitrogen', 'Phosphorus', 'Potassium']
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    elif user_type == 'researcher':
        tabs = st.tabs(['📊 Data Overview', '🤖 Model Performance', '🗺️ Spatial Analysis'])
        
        with tabs[0]:
            st.header("📈 Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Summary")
                st.dataframe(df.describe())
            
            with col2:
                st.subheader("Correlation Matrix")
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # State-wise analysis
            st.subheader("🌍 State-wise Nutrient Limitations")
            state_stats = df.groupby('State')[['N_Limited', 'P_Limited', 'K_Limited']].mean()
            
            fig = px.bar(
                state_stats,
                title="Average Nutrient Limitations by State",
                labels={'value': 'Limitation Frequency', 'index': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.header("🎯 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            for i, (nutrient, score) in enumerate(scores.items()):
                with [col1, col2, col3][i]:
                    st.metric(
                        f"{nutrient} Model Accuracy",
                        f"{score:.3f}",
                        f"{'✅ Good' if score > 0.8 else '⚠️ Fair' if score > 0.7 else '❌ Poor'}"
                    )
            
            # Feature importance
            st.subheader("🔍 Feature Importance Analysis")
            
            importance_data = []
            for nutrient, model in models.items():
                for feature, importance in zip(features, model.feature_importances_):
                    importance_data.append({
                        'Nutrient': nutrient,
                        'Feature': feature,
                        'Importance': importance
                    })
            
            importance_df = pd.DataFrame(importance_data)
            
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                color='Nutrient',
                title="Feature Importance by Nutrient Model",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.header("🗺️ Spatial Distribution Analysis")
            
            # Simulated spatial analysis results
            st.subheader("📍 Spatial Autocorrelation (Moran's I)")
            
            col1, col2, col3 = st.columns(3)
            
            # Simulated Moran's I values
            morans_i = {'N': 0.156, 'P': 0.423, 'K': 0.089}
            
            with col1:
                st.metric("Nitrogen Moran's I", f"{morans_i['N']:.3f}", "Weak positive")
            with col2:
                st.metric("Phosphorus Moran's I", f"{morans_i['P']:.3f}", "Moderate positive")
            with col3:
                st.metric("Potassium Moran's I", f"{morans_i['K']:.3f}", "Very weak")
            
            # Moran's I visualization
            fig = px.bar(
                x=list(morans_i.keys()),
                y=list(morans_i.values()),
                title="Spatial Autocorrelation (Moran's I) by Nutrient",
                labels={'x': 'Nutrient', 'y': "Moran's I Value"},
                color=list(morans_i.values()),
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive Maps
            st.subheader("🗺️ Interactive Nutrient Maps")
            
            # Map options
            map_type = st.radio(
                "Select Map Type:",
                ["Individual LGA Analysis", "State Summary Overview"],
                horizontal=True
            )
            
            if map_type == "Individual LGA Analysis":
                nutrient_choice = st.selectbox(
                    "Select nutrient to visualize", 
                    ['N_Limited', 'P_Limited', 'K_Limited'],
                    format_func=lambda x: x.replace('_Limited', ' Limitation')
                )
                
                st.write(f"**{nutrient_choice.replace('_', ' ')} across Nigerian LGAs**")
                
                # Create interactive map
                nigeria_map = create_nigeria_map(
                    df, 
                    nutrient_choice, 
                    f"{nutrient_choice.replace('_', ' ')}"
                )
                
                # Display map
                st_folium(nigeria_map, width=700, height=500)
                
                # Add interpretation
                limitation_pct = df[nutrient_choice].mean() * 100
                limited_lgas = df[df[nutrient_choice] == True]['LGA'].tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"LGAs with {nutrient_choice.replace('_Limited', '')} Limitation", 
                        f"{len(limited_lgas)} out of {len(df)}"
                    )
                with col2:
                    st.metric("Limitation Percentage", f"{limitation_pct:.1f}%")
                
                if limited_lgas:
                    st.write("**LGAs requiring attention:**")
                    for lga in limited_lgas[:5]:  # Show first 5
                        st.write(f"• {lga}")
                    if len(limited_lgas) > 5:
                        st.write(f"... and {len(limited_lgas) - 5} more")
            
            else:  # State Summary Overview
                st.write("**State-level Nutrient Risk Summary**")
                
                # Create state summary map
                state_map = create_state_summary_map(df)
                
                # Display map
                st_folium(state_map, width=700, height=500)
                
                # Add state comparison table
                st.subheader("📊 State Comparison Table")
                state_stats = df.groupby('State').agg({
                    'N_Limited': 'mean',
                    'P_Limited': 'mean', 
                    'K_Limited': 'mean'
                }).round(3)
                
                state_stats.columns = ['Nitrogen Risk', 'Phosphorus Risk', 'Potassium Risk']
                state_stats = state_stats.sort_values('Phosphorus Risk', ascending=False)
                
                st.dataframe(state_stats, use_container_width=True)
    
    # Extension and policy interfaces (simplified)
    elif user_type == 'extension':
        st.header("🎓 Extension Agent Dashboard")
        
        selected_state = st.selectbox("Select State for Analysis", df['State'].unique())
        state_data = df[df['State'] == selected_state]
        
        if not state_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_pct = state_data['N_Limited'].mean() * 100
                st.metric("N Limitation", f"{n_pct:.1f}%")
            with col2:
                p_pct = state_data['P_Limited'].mean() * 100
                st.metric("P Limitation", f"{p_pct:.1f}%")
            with col3:
                k_pct = state_data['K_Limited'].mean() * 100
                st.metric("K Limitation", f"{k_pct:.1f}%")
            
            st.dataframe(state_data[['LGA', 'N_Limited', 'P_Limited', 'K_Limited']])
    
    elif user_type == 'policy':
        st.header("🏛️ Policy Maker Dashboard")
        
        # National overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            national_n = df['N_Limited'].mean() * 100
            st.metric("National N Limitation", f"{national_n:.1f}%")
        with col2:
            national_p = df['P_Limited'].mean() * 100
            st.metric("National P Limitation", f"{national_p:.1f}%")
        with col3:
            national_k = df['K_Limited'].mean() * 100
            st.metric("National K Limitation", f"{national_k:.1f}%")
        
        # State comparison
        state_comparison = df.groupby('State')[['N_Limited', 'P_Limited', 'K_Limited']].mean()
        
        fig = px.bar(
            state_comparison,
            title="State-wise Nutrient Limitation Comparison",
            labels={'value': 'Limitation Frequency', 'index': 'State'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Nutrient Limitation Strategies & Spatial Analysis** | "
        "Advanced agricultural decision support for Nigerian farmers | "
        f"Model Performance: N({scores['N']:.2f}), P({scores['P']:.2f}), K({scores['K']:.2f})"
    )

if __name__ == "__main__":
    main()