# 🌱 Nutrient Limitation Strategies & Spatial Analysis

Advanced spatial analysis tools for agricultural nutrient management in Nigeria, featuring Moran's I analysis, LISA clustering, and interactive mapping.

## 🎯 Project Overview

This project provides comprehensive spatial analysis for understanding nutrient limitations in agricultural systems, with a focus on:

- **Spatial Autocorrelation Analysis**: Using Moran's I to identify spatial patterns
- **Local Indicators of Spatial Association (LISA)**: Detecting local clusters and outliers  
- **Interactive Mapping**: Visualizing nutrient distributions and limitations
- **Multi-language Support**: English and Hausa interfaces
- **User-specific Tools**: Tailored for farmers, researchers, extension agents, and policy makers

## 🔬 Key Analysis Methods

### Spatial Statistics
- **Moran's I**: Global spatial autocorrelation measurement
- **LISA Analysis**: Local cluster and outlier identification
- **Spatial Weight Matrices**: Queen and Rook contiguity
- **Hotspot Analysis**: Getis-Ord Gi* statistics

### Machine Learning
- **Random Forest Classification**: Nutrient limitation prediction
- **Feature Importance Analysis**: Understanding key predictors
- **Cross-validation**: Model performance assessment

### Visualization
- **Interactive Maps**: Folium-based mapping with real LGA boundaries
- **Statistical Charts**: Plotly visualizations for spatial statistics
- **Correlation Analysis**: Heatmaps and relationship plots

## 📊 Data Structure

### Input Data Format
```csv
Latitude,Longitude,N_ppm,P_ppm,K_ppm,pH,Organic_Matter,LGA,State
12.0123,8.5234,25.3,12.1,145.2,6.5,2.3,Kano Municipal,Kano
```

### Analysis Outputs
- Spatial autocorrelation indices
- LISA cluster classifications
- Nutrient limitation probabilities
- Interactive maps with statistical overlays

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/NutrientLimitationStrategies.git
cd NutrientLimitationStrategies

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the application
streamlit run streamlit_app/main.py
```

### Data Preparation
1. Ensure soil test data includes geographic coordinates
2. Include nutrient measurements (N, P, K in ppm)
3. Add soil properties (pH, organic matter percentage)
4. Include administrative boundaries (LGA, State)

## 📁 Project Structure

```
NutrientLimitationStrategies/
├── notebooks/
│   └── 01_nutrient_limitation_analysis.ipynb  # Complete analysis workflow
├── data/
│   ├── raw/                                   # Original datasets
│   ├── processed/                             # Cleaned and prepared data
│   └── boundaries/                            # LGA boundary files
├── streamlit_app/
│   ├── main.py                               # Main application
│   ├── requirements.txt                      # Dependencies
│   └── utils/                                # Helper functions
└── models/
    └── trained_models/                       # Saved ML models
```

## 🗺️ Spatial Analysis Features

### Global Spatial Statistics
- **Moran's I**: Measures overall spatial clustering
- **Geary's C**: Alternative autocorrelation measure
- **Join Count Statistics**: For categorical data analysis

### Local Spatial Statistics
- **Local Moran's I**: Identifies local clusters (LISA)
- **Getis-Ord Gi***: Hotspot and coldspot detection
- **Local Geary**: Local spatial heterogeneity

### Cluster Types
- **High-High**: Areas with high values surrounded by high values
- **Low-Low**: Areas with low values surrounded by low values  
- **High-Low**: Spatial outliers (high surrounded by low)
- **Low-High**: Spatial outliers (low surrounded by high)

## 👥 User Interfaces

### 🌱 Farmers
- Simple nutrient status checking
- Local language support (Hausa)
- Easy-to-understand recommendations

### 🔬 Researchers
- Full spatial analysis toolkit
- Advanced statistical outputs
- Model performance metrics
- Data export capabilities

### 🎓 Extension Agents
- Area-specific analysis tools
- Bulk processing capabilities
- Educational visualizations
- Recommendation summaries

### 🏛️ Policy Makers
- Regional trend analysis
- Resource allocation insights
- State-level comparisons
- Policy impact assessments

## 📈 Analysis Results

### Spatial Patterns Discovered
- **Nitrogen**: Widespread limitations (87.5% of areas)
- **Phosphorus**: Strong spatial clustering in specific regions
- **Potassium**: Random distribution patterns
- **pH**: Regional trends affecting nutrient availability

### Key Findings
- Phosphorus shows strongest spatial autocorrelation (Moran's I = 0.423)
- Hotspot clusters identified in specific LGAs
- Soil pH significantly influences nutrient availability patterns

## 🛠️ Technology Stack

- **Backend**: Python, pandas, numpy, scikit-learn
- **Spatial Analysis**: pysal, libpysal, esda, splot
- **Geospatial**: geopandas, folium
- **Visualization**: plotly, matplotlib, seaborn
- **Web Framework**: Streamlit
- **Data**: Nigerian GADM boundaries, soil test data

## 📖 Documentation

- **Notebook**: Complete analysis in `notebooks/01_nutrient_limitation_analysis.ipynb`
- **API Reference**: Function documentation in source code
- **User Guide**: Built-in help system in web application
- **Methodology**: Spatial statistics theory and implementation notes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/spatial-analysis`)
3. Commit your changes (`git commit -m 'Add spatial clustering analysis'`)
4. Push to the branch (`git push origin feature/spatial-analysis`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions about spatial analysis methods or implementation:
- Email: spatial.analysis@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Nigerian agricultural research institutions
- GADM for administrative boundary data
- PySAL community for spatial analysis tools
- Nigerian farmers and extension agents for domain expertise