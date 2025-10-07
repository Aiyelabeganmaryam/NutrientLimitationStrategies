import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
def generate_sample_data():
    """Generate sample agricultural data for demonstration"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate realistic agricultural data
    data = {
        'Latitude': np.random.uniform(8.5, 13.5, n_samples),
        'Longitude': np.random.uniform(3.0, 15.0, n_samples),
        'State': np.random.choice(['Kano', 'Kaduna', 'Katsina', 'Jigawa', 'Bauchi', 'Sokoto'], n_samples),
        'LGA': [f'LGA_{i:03d}' for i in range(n_samples)],
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
            
            # Geographic distribution
            st.subheader("🌍 Geographic Distribution")
            
            nutrient_choice = st.selectbox("Select nutrient to visualize", ['N_Limited', 'P_Limited', 'K_Limited'])
            
            fig = px.scatter(
                df,
                x='Longitude',
                y='Latitude',
                color=nutrient_choice,
                title=f"{nutrient_choice.replace('_', ' ')} Distribution",
                hover_data=['State', 'LGA'],
                color_discrete_map={True: 'red', False: 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
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