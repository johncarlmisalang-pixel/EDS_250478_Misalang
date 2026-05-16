# EDS_250351_Inocencio
**Engineering Data Systems Pipeline — ENV-02: Aerosol Optical Depth vs. Humidity**  
Course: Computer Programming 1 | Academic Year: 2026  
Student: Marv Andrew S. Inocencio | ID: TUPM-25-0351

---

## Topic Assignment
| Field | Value |
|-------|-------|
| Student Number Last Digit | 1 → Pillar 1: Environmental Engineering |
| Surname First Letter | I → F–J → Topic Concept 02 |
| **Assigned Topic** | **ENV-02: Aerosol Optical Depth vs. Humidity** |
| Dataset | Global Air Quality Data (Kaggle) |
| Unique Filter | `Country == "India"` (540 records) |

## Project Structure
```
EDS_250351_Inocencio/
├── main.py                  # Full OOP data pipeline
├── requirements.txt         # Required Python libraries
├── README.md                # This file
├── data/
│   ├── dataset_original.csv # India-filtered raw dataset
│   └── dataset_cleaned.csv  # Cleaned dataset with derived columns
└── outputs/
    ├── plot1_histogram.png                  # Aerosol Proxy distribution
    ├── plot2_boxplot.png                    # Low vs High Humidity groups
    ├── plot3_heatmap.png                    # Pearson correlation heatmap
    ├── plot4_scatter_humidity_aerosol.png   # Humidity vs Aerosol scatter
    ├── animation1_rolling_mean.gif          # Animated rolling mean
    └── animation2_scatter_cluster.gif       # Animated cluster scatter
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the original CSV as `data/dataset_original.csv`  
   *(or use the one already included — it is the India-filtered slice)*
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. All outputs are saved to the `outputs/` folder automatically.

## Pipeline Modules
| # | Method | Description |
|---|--------|-------------|
| 1 | `ingest_data()` | Loads CSV, applies `Country == "India"` unique filter |
| 2 | `clean_data()` | Removes duplicates/nulls, coerces types, computes Aerosol Proxy Index & Hygroscopic Score |
| 3 | `analyze_data()` | NumPy descriptive stats, skewness, IQR outlier detection, Pearson correlation, Low vs High Humidity group comparison |
| 4 | `visualize_static()` | Histogram, Boxplot, Correlation Heatmap, Humidity vs Aerosol scatter |
| 5 | `visualize_animated()` | Rolling mean GIF, Humidity vs Hygro Score animated scatter GIF |

## Derived Variables
- **Aerosol_Proxy** = `0.6 × PM2.5 + 0.4 × PM10` — ground-level proxy for Aerosol Optical Depth
- **Hygro_Score** = `Aerosol_Proxy × (Humidity / 100)` — models hygroscopic aerosol swelling effect
