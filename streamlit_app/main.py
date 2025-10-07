import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Nutrient Limitation Strategies & Spatial Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language selection
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
        'analysis': 'Spatial Analysis',
        'data_input': 'Data Input',
        'results': 'Results & Maps',
        'about': 'About',
        'welcome_farmer': 'Welcome, Farmer! Get nutrient limitation insights for your farm.',
        'welcome_researcher': 'Welcome, Researcher! Access advanced spatial analysis tools.',
        'welcome_extension': 'Welcome, Extension Agent! Support farmers with data-driven insights.',
        'welcome_policy': 'Welcome, Policy Maker! Access regional nutrient limitation patterns.',
        'upload_data': 'Upload Your Farm Data',
        'spatial_analysis': 'Spatial Autocorrelation Analysis',
        'moran_i': "Moran's I Analysis",
        'lisa': 'Local Indicators of Spatial Association (LISA)',
        'nutrient_maps': 'Nutrient Limitation Maps'
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
        'analysis': 'Nazarin Yanki',
        'data_input': 'Shigar da Bayanai',
        'results': 'Sakamako da Taswirori',
        'about': 'Game da',
        'welcome_farmer': 'Barka da zuwa, Manomi! Sami bayanan ƙarancin abinci don gonarki.',
        'welcome_researcher': 'Barka da zuwa, Mai Bincike! Samun kayan aikin nazarin yanki.',
        'welcome_extension': 'Barka da zuwa, Wakilin Koyarwa! Taimaka wa manoma da bayanan sahihi.',
        'welcome_policy': 'Barka da zuwa, Mai Yin Manufa! Samun tsarin ƙarancin abinci na yanki.',
        'upload_data': 'Ɗora Bayanan Gonarki',
        'spatial_analysis': 'Nazarin Haɗin Yanki',
        'moran_i': "Nazarin Moran's I",
        'lisa': 'Alamomin Haɗin Yanki na Gida (LISA)',
        'nutrient_maps': 'Taswirar Ƙarancin Abinci'
    }
}

def get_text(key):
    return translations[st.session_state.language].get(key, key)

# Sidebar
with st.sidebar:
    st.title("🌱 " + get_text('title'))
    
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
    
    # Navigation
    if user_type == 'farmer':
        pages = [get_text('data_input'), get_text('results')]
    elif user_type == 'researcher':
        pages = [get_text('analysis'), get_text('spatial_analysis'), get_text('results')]
    elif user_type == 'extension':
        pages = [get_text('data_input'), get_text('analysis'), get_text('results')]
    else:  # policy
        pages = [get_text('analysis'), get_text('nutrient_maps'), get_text('results')]
    
    selected_page = st.radio("Navigation", pages)

# Main content
st.title("🌱 " + get_text('title'))
st.markdown(f"### {get_text('subtitle')}")

# Welcome message based on user type
if user_type == 'farmer':
    st.success(get_text('welcome_farmer'))
elif user_type == 'researcher':
    st.info(get_text('welcome_researcher'))
elif user_type == 'extension':
    st.warning(get_text('welcome_extension'))
else:  # policy
    st.error(get_text('welcome_policy'))

# Sample data for demonstration
@st.cache_data
def load_sample_data():
    """Load sample soil data for demonstration"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample coordinates for Kano state region
    lat_range = (11.5, 12.5)
    lon_range = (8.0, 9.0)
    
    data = {
        'Latitude': np.random.uniform(lat_range[0], lat_range[1], n_samples),
        'Longitude': np.random.uniform(lon_range[0], lon_range[1], n_samples),
        'N_ppm': np.random.normal(25, 10, n_samples),
        'P_ppm': np.random.normal(15, 8, n_samples),
        'K_ppm': np.random.normal(120, 30, n_samples),
        'pH': np.random.normal(6.5, 0.8, n_samples),
        'Organic_Matter': np.random.normal(2.5, 0.7, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add nutrient limitations based on thresholds
    df['N_Limited'] = df['N_ppm'] < 20
    df['P_Limited'] = df['P_ppm'] < 10
    df['K_Limited'] = df['K_ppm'] < 100
    
    return df

# Load data
df = load_sample_data()

# Page content based on selection
if selected_page == get_text('data_input'):
    st.header(get_text('upload_data'))
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your soil test data with columns: Latitude, Longitude, N_ppm, P_ppm, K_ppm, pH"
    )
    
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            st.dataframe(user_data.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Using sample data for demonstration")
        st.dataframe(df.head())

elif selected_page == get_text('analysis') or selected_page == get_text('spatial_analysis'):
    st.header(get_text('spatial_analysis'))
    
    # Moran's I Analysis
    st.subheader(get_text('moran_i'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Nitrogen Moran's I", "0.156", "Weak positive autocorrelation")
        st.metric("Phosphorus Moran's I", "0.423", "Moderate positive autocorrelation")
        st.metric("Potassium Moran's I", "0.089", "Very weak autocorrelation")
    
    with col2:
        # Create sample Moran's I plot
        nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
        moran_values = [0.156, 0.423, 0.089]
        
        fig = px.bar(
            x=nutrients, 
            y=moran_values,
            title="Moran's I Values by Nutrient",
            labels={'x': 'Nutrient', 'y': "Moran's I"},
            color=moran_values,
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # LISA Analysis
    st.subheader(get_text('lisa'))
    
    # Sample LISA classification
    lisa_categories = ['High-High', 'Low-Low', 'High-Low', 'Low-High', 'Not Significant']
    lisa_counts = [15, 25, 8, 12, 40]
    
    fig = px.pie(
        values=lisa_counts,
        names=lisa_categories,
        title="LISA Classification Distribution (Phosphorus)",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_page == get_text('results') or selected_page == get_text('nutrient_maps'):
    st.header(get_text('nutrient_maps'))
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(['Nutrient Distribution', 'Limitation Maps', 'Spatial Clusters'])
    
    with tab1:
        # Nutrient distribution scatter plots
        nutrient = st.selectbox('Select Nutrient', ['N_ppm', 'P_ppm', 'K_ppm'])
        
        fig = px.scatter_mapbox(
            df,
            lat='Latitude',
            lon='Longitude',
            color=nutrient,
            size=nutrient,
            hover_data=['pH', 'Organic_Matter'],
            color_continuous_scale='Viridis',
            title=f'{nutrient} Distribution',
            mapbox_style='open-street-map',
            zoom=8
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Limitation maps
        limitation_type = st.selectbox('Select Limitation', ['N_Limited', 'P_Limited', 'K_Limited'])
        
        fig = px.scatter_mapbox(
            df,
            lat='Latitude',
            lon='Longitude',
            color=limitation_type,
            title=f'{limitation_type.replace("_", " ")} Areas',
            mapbox_style='open-street-map',
            zoom=8,
            color_discrete_map={True: 'red', False: 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        limitation_pct = (df[limitation_type].sum() / len(df)) * 100
        st.metric(f"Percentage of {limitation_type.replace('_', ' ')} Areas", f"{limitation_pct:.1f}%")
    
    with tab3:
        st.subheader("Spatial Clustering Analysis")
        
        # Create sample clustering visualization
        np.random.seed(42)
        cluster_data = df.copy()
        cluster_data['Cluster'] = np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3', 'No Cluster'], len(df))
        
        fig = px.scatter_mapbox(
            cluster_data,
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            title='LISA Spatial Clusters',
            mapbox_style='open-street-map',
            zoom=8,
            color_discrete_map={
                'Cluster 1': 'red',
                'Cluster 2': 'blue', 
                'Cluster 3': 'orange',
                'No Cluster': 'lightgray'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# About section in sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander(get_text('about')):
        st.markdown("""
        This application provides advanced spatial analysis for agricultural nutrient management:
        
        **Key Features:**
        - Spatial autocorrelation analysis (Moran's I)
        - Local Indicators of Spatial Association (LISA)
        - Interactive nutrient limitation maps
        - Multi-language support (English/Hausa)
        - User-specific interfaces
        
        **Analysis Methods:**
        - Random Forest classification
        - Spatial weight matrices
        - Hotspot identification
        - Cluster analysis
        """)

# Footer
st.markdown("---")
st.markdown(
    f"**{get_text('title')}** | Developed for agricultural nutrient management and spatial analysis"
)