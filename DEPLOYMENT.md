# Deployment Guide: Nutrient Limitation Strategies & Spatial Analysis

## Project Overview
This project has been renamed from "FertilizerRecommendationApp" to "NutrientLimitationStrategies" to better reflect its focus on spatial analysis, Moran's I statistics, LISA clustering, and nutrient limitation mapping.

## Quick Setup

### Prerequisites
- Python 3.8 or higher
- Git (for version control)
- Web browser for Streamlit interface

### 1. Local Development Setup

```bash
# Navigate to the project directory
cd C:\Users\DELL\NutrientLimitationStrategies

# Create virtual environment
python -m venv nutrient_env

# Activate virtual environment (Windows)
nutrient_env\Scripts\activate

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the application
streamlit run streamlit_app/main.py
```

### 2. Streamlit Cloud Deployment

1. **Create GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Nutrient Limitation Strategies & Spatial Analysis"
   git branch -M main
   git remote add origin https://github.com/yourusername/NutrientLimitationStrategies.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository: `NutrientLimitationStrategies`
   - Set main file path: `streamlit_app/main.py`
   - Click "Deploy"

### 3. Alternative Cloud Deployment Options

#### Heroku
```bash
# Install Heroku CLI
# Create Procfile in streamlit_app directory
echo "web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0" > streamlit_app/Procfile

# Deploy
heroku create nutrient-limitation-strategies
git push heroku main
```

#### Railway
```bash
# Create railway.toml
echo "[build]
command = \"pip install -r requirements.txt\"
[deploy]
startCommand = \"streamlit run main.py --server.port=$PORT --server.address=0.0.0.0\"" > streamlit_app/railway.toml
```

## Project Structure

```
NutrientLimitationStrategies/
├── README.md                          # Project overview and documentation
├── notebooks/
│   └── 01_nutrient_limitation_analysis.ipynb  # Complete spatial analysis
├── data/
│   ├── raw/                          # Original datasets
│   ├── processed/                    # Cleaned data
│   └── boundaries/                   # Nigerian LGA boundaries
├── streamlit_app/
│   ├── main.py                       # Main application
│   ├── requirements.txt              # Python dependencies
│   └── utils/                        # Helper functions
└── models/
    └── trained_models/               # Saved ML models
```

## Key Features

### Spatial Analysis Capabilities
- **Moran's I Analysis**: Global spatial autocorrelation
- **LISA Clustering**: Local Indicators of Spatial Association
- **Hotspot Detection**: Getis-Ord Gi* statistics
- **Interactive Mapping**: Real Nigerian LGA boundaries

### User Interfaces
- **Farmers**: Simple nutrient checking (English/Hausa)
- **Researchers**: Full spatial analysis toolkit
- **Extension Agents**: Area-specific recommendations
- **Policy Makers**: Regional trend analysis

### Machine Learning
- **Random Forest Models**: Nutrient limitation prediction
- **Feature Importance**: Understanding key predictors
- **Spatial Validation**: Cross-validation with spatial awareness

## Configuration

### Environment Variables (Optional)
Create `.env` file in streamlit_app directory:
```env
MAPBOX_ACCESS_TOKEN=your_token_here
DATABASE_URL=your_database_url_here
DEBUG_MODE=false
```

### Customization Options
1. **Language Support**: Extend translations in `main.py`
2. **Analysis Parameters**: Modify spatial weight matrices
3. **Visualization**: Customize map styles and colors
4. **Data Sources**: Connect to external databases

## Data Requirements

### Input Format
```csv
Latitude,Longitude,N_ppm,P_ppm,K_ppm,pH,Organic_Matter,LGA,State
12.0123,8.5234,25.3,12.1,145.2,6.5,2.3,Kano Municipal,Kano
11.9876,8.4567,18.7,8.9,132.1,6.8,1.9,Gwale,Kano
```

### Boundary Data
- Nigerian LGA boundaries (GeoJSON format)
- Available from GADM database
- Included in project data directory

## Performance Optimization

### Caching Strategies
```python
@st.cache_data
def load_spatial_data():
    return gpd.read_file('data/boundaries/nigeria_lgas.geojson')

@st.cache_resource
def train_spatial_models():
    return trained_models
```

### Memory Management
- Use data streaming for large datasets
- Implement lazy loading for boundary files
- Cache spatial weight matrices

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Spatial Analysis Errors**:
   - Check coordinate system (WGS84 recommended)
   - Verify spatial weight matrix connectivity
   - Ensure sufficient sample size for statistics

3. **Memory Issues**:
   - Reduce sample size for development
   - Use spatial sampling techniques
   - Implement data chunking

### Performance Tips
- Enable Streamlit caching for data loading
- Use efficient spatial indexing
- Optimize coordinate transformations

## Security Considerations

### Production Deployment
- Use environment variables for sensitive data
- Implement input validation
- Enable HTTPS in production
- Monitor resource usage

### Data Privacy
- Anonymize location data if required
- Implement user access controls
- Regular security audits

## Monitoring and Maintenance

### Application Monitoring
- Track user interactions
- Monitor spatial analysis performance
- Log error rates and types

### Data Updates
- Regular boundary data updates
- Model retraining schedules
- Validation of spatial statistics

## Support and Documentation

### Getting Help
- GitHub Issues: Report bugs and feature requests
- Documentation: Built-in help system in app
- Email: spatial.analysis@example.com

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/spatial-enhancement`)
3. Commit changes (`git commit -m 'Add spatial clustering'`)
4. Push to branch (`git push origin feature/spatial-enhancement`)
5. Create Pull Request

## Future Enhancements

### Planned Features
- **Time Series Analysis**: Temporal spatial patterns
- **Advanced Clustering**: DBSCAN, K-means spatial
- **Machine Learning**: Deep learning for spatial prediction
- **API Development**: REST API for external integration

### Research Integration
- Academic collaboration opportunities
- Publication of spatial methods
- Conference presentations
- Open dataset contributions

---

**Project Focus**: Spatial autocorrelation, LISA clustering, and nutrient limitation strategies
**Technology**: Python, Streamlit, PySAL, Plotly, Folium
**Target Users**: Agricultural researchers, extension agents, policy makers, farmers