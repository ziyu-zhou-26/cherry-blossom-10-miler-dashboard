import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def load_data():
    """Load the raw scraped data."""
    logging.info("Loading raw data...")
    df = pd.read_csv(RAW_DATA_DIR / "CherryBlossom2025_final.csv")
    logging.info(f"Loaded {len(df)} records")
    
    # Print initial info
    print("\nInitial Data Preview:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nData types:", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    
    return df

def time_to_seconds(time_str):
    """Convert 'H:MM:SS' or 'MM:SS' to total seconds."""
    if pd.isna(time_str) or time_str == '':
        return np.nan
    try:
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        
        if len(parts) == 3:  # H:MM:SS
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:  # MM:SS
            m, s = parts
            return int(m) * 60 + int(s)
        else:
            return np.nan
    except:
        return np.nan

def clean_data(df):
    """Clean and transform the data."""
    logging.info("\nCleaning data...")
    df_clean = df.copy()
    
    # 1. Clean column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Convert Gender to standardized format
    df_clean['Gender'] = df_clean['Gender'].str.strip().str.upper()
    logging.info(f"Gender distribution: {df_clean['Gender'].value_counts().to_dict()}")
    
    # 3. Convert Age to numeric
    df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
    logging.info(f"Age range: {df_clean['Age'].min()} to {df_clean['Age'].max()}")
    logging.info(f"Missing ages: {df_clean['Age'].isnull().sum()}")
    
    # 4. Clean State - handle empty strings
    df_clean['State'] = df_clean['State'].replace('', np.nan)
    df_clean['State'] = df_clean['State'].str.strip().str.upper()
    
    # 5. Clean Country
    df_clean['Country'] = df_clean['Country'].str.strip().str.upper()
    logging.info(f"Countries represented: {df_clean['Country'].nunique()}")
    
    # 6. Create US vs International flag
    df_clean['Is_US'] = df_clean['Country'] == 'USA'
    logging.info(f"US runners: {df_clean['Is_US'].sum()}")
    logging.info(f"US States represented: {df_clean[df_clean['Is_US']]['State'].nunique()}")
    # Remove military postal codes
    military_codes = ['AA', 'AE', 'AP']
    df_clean = df_clean[~df_clean['State'].isin(military_codes)]
    
    # 7. Convert place columns to numeric
    place_columns = ['Overall Place', 'Gender Place', 'Age Group Place']
    for col in place_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 8. Convert Finish Time to seconds
    df_clean['Finish Time (seconds)'] = df_clean['Finish Time'].apply(time_to_seconds)
    logging.info(f"Finish time conversion - Missing: {df_clean['Finish Time (seconds)'].isnull().sum()}")
    
    # 9. Convert Finish Time to minutes (easier to read)
    df_clean['Finish Time (minutes)'] = df_clean['Finish Time (seconds)'] / 60
    
    # 10. Convert Pace to seconds per mile
    df_clean['Pace (sec/mile)'] = df_clean['Pace'].apply(time_to_seconds)
    df_clean['Pace (min/mile)'] = df_clean['Pace (sec/mile)'] / 60
    
    # 11. Create Age Groups
    bins = [0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 100]
    labels = ['0-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
          '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    df_clean['Age Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
    logging.info(f"Age group distribution:\n{df_clean['Age Group'].value_counts().sort_index()}")
    
    # 12. Remove records with missing critical data
    # only require finish time and pace
    before_removal = len(df_clean)
    df_clean = df_clean.dropna(subset=['Finish Time (seconds)', 'Pace (sec/mile)'])
    removed = before_removal - len(df_clean)
    logging.info(f"Removed {removed} records with missing finish time or pace")
    logging.info(f"Removed {removed} records with missing critical data")
    
    # 13. Remove outliers (unrealistic times for 10-mile race)
    min_time_seconds = 40 * 60  # 40 minutes
    max_time_seconds = 180 * 60  # 3 hours
    
    before_outlier = len(df_clean)
    df_clean = df_clean[
        (df_clean['Finish Time (seconds)'] >= min_time_seconds) & 
        (df_clean['Finish Time (seconds)'] <= max_time_seconds)
    ]
    removed_outliers = before_outlier - len(df_clean)
    logging.info(f"Removed {removed_outliers} outliers (times outside 40min-3hr)")
    
    # 14. Create US Census Bureau Regions and Divisions
    # Source: https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf
    
    # Northeast Region
    new_england = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']
    middle_atlantic = ['NJ', 'NY', 'PA']
    
    # Midwest Region
    east_north_central = ['IL', 'IN', 'MI', 'OH', 'WI']
    west_north_central = ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
    
    # South Region
    south_atlantic = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV']
    east_south_central = ['AL', 'KY', 'MS', 'TN']
    west_south_central = ['AR', 'LA', 'OK', 'TX']
    
    # West Region
    mountain = ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY']
    pacific = ['AK', 'CA', 'HI', 'OR', 'WA']
    
    def get_census_region(state):
        if pd.isna(state):
            return np.nan
        state = str(state).upper().strip()
        
        # Handle military postal codes
        if state in ['AA', 'AE', 'AP']:
            return 'U.S. Military'
        
        if state in new_england or state in middle_atlantic:
            return 'Northeast'
        elif state in east_north_central or state in west_north_central:
            return 'Midwest'
        elif state in south_atlantic or state in east_south_central or state in west_south_central:
            return 'South'
        elif state in mountain or state in pacific:
            return 'West'
        else:
            return 'Other'

    def get_census_division(state):
        if pd.isna(state):
            return np.nan
        state = str(state).upper().strip()
        
        # Handle military postal codes
        if state in ['AA', 'AE', 'AP']:
            return 'U.S. Military'
        
        if state in new_england:
            return 'New England'
        elif state in middle_atlantic:
            return 'Middle Atlantic'
        elif state in east_north_central:
            return 'East North Central'
        elif state in west_north_central:
            return 'West North Central'
        elif state in south_atlantic:
            return 'South Atlantic'
        elif state in east_south_central:
            return 'East South Central'
        elif state in west_south_central:
            return 'West South Central'
        elif state in mountain:
            return 'Mountain'
        elif state in pacific:
            return 'Pacific'
        else:
            return 'Other'
    
    df_clean['Census Region'] = df_clean['State'].apply(get_census_region)
    df_clean['Census Division'] = df_clean['State'].apply(get_census_division)
    
    # 15. Flag local runners (DC, MD, VA)
    df_clean['Is_Local'] = df_clean['State'].isin(['DC', 'MD', 'VA'])
    
    return df_clean

def calculate_percentiles(df):
    """Calculate percentile ranks for each runner."""
    logging.info("\nCalculating percentiles...")
    
    # Overall percentile
    df['Overall Percentile'] = df['Overall Place'].rank(pct=True) * 100
    
    # Gender-specific percentile
    df['Gender Percentile'] = df.groupby('Gender')['Gender Place'].rank(pct=True) * 100
    
    # Age group percentile
    df['Age Group Percentile'] = df.groupby(['Gender', 'Age Group'])['Age Group Place'].rank(pct=True) * 100
    
    return df

def generate_summary_stats(df):
    """Generate summary statistics."""
    logging.info("\n" + "="*60)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*60)
    
    print(f"\nTotal Runners: {len(df)}")
    print(f"Male: {len(df[df['Gender'] == 'M'])} ({len(df[df['Gender'] == 'M'])/len(df)*100:.1f}%)")
    print(f"Female: {len(df[df['Gender'] == 'F'])} ({len(df[df['Gender'] == 'F'])/len(df)*100:.1f}%)")
    
    print(f"\nAge Statistics:")
    print(f"  Average: {df['Age'].mean():.1f} years")
    print(f"  Median: {df['Age'].median():.0f} years")
    print(f"  Range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")
    
    print(f"\nFinish Time Statistics:")
    print(f"  Average: {df['Finish Time (minutes)'].mean():.1f} minutes")
    print(f"  Median: {df['Finish Time (minutes)'].median():.1f} minutes")
    print(f"  Fastest: {df['Finish Time (minutes)'].min():.1f} minutes")
    print(f"  Slowest: {df['Finish Time (minutes)'].max():.1f} minutes")
    
    print(f"\nPace Statistics:")
    print(f"  Average: {df['Pace (min/mile)'].mean():.2f} min/mile")
    print(f"  Median: {df['Pace (min/mile)'].median():.2f} min/mile")
    
    print(f"\nGeographic Distribution:")
    print(f"  US Runners: {df['Is_US'].sum()} ({df['Is_US'].sum()/len(df)*100:.1f}%)")
    print(f"  International: {(~df['Is_US']).sum()} ({(~df['Is_US']).sum()/len(df)*100:.1f}%)")
    print(f"  Local (DC/MD/VA): {df['Is_Local'].sum()} ({df['Is_Local'].sum()/len(df)*100:.1f}%)")
    print(f"  Unique States: {df['State'].nunique()}")
    print(f"  Unique Countries: {df['Country'].nunique()}")
    
    print(f"\nTop 10 States by Participation:")
    print(df[df['Is_US']]['State'].value_counts().head(10))
    
    print(f"\nUS Census Region Distribution:")
    print(df[df['Is_US']]['Census Region'].value_counts())
    
    print(f"\nUS Census Division Distribution:")
    print(df[df['Is_US']]['Census Division'].value_counts())
    
    print(f"\nAverage Finish Time by Gender:")
    print(df.groupby('Gender')['Finish Time (minutes)'].mean())
    
    print(f"\nAverage Finish Time by Age Group:")
    print(df.groupby('Age Group')['Finish Time (minutes)'].mean().sort_index())
    
    print(f"\nAverage Finish Time by Census Region:")
    print(df[df['Is_US']].groupby('Census Region')['Finish Time (minutes)'].mean().sort_values())

def save_cleaned_data(df):
    """Save the cleaned dataset."""
    output_file = PROCESSED_DATA_DIR / "cherry_blossom_2025_cleaned.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"\n✓ Cleaned data saved to: {output_file}")
    logging.info(f"✓ Final dataset: {len(df)} records, {len(df.columns)} columns")
    
    # Also save a data dictionary
    data_dict = {
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Sample Value': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
    }
    dict_df = pd.DataFrame(data_dict)
    dict_file = PROCESSED_DATA_DIR / "data_dictionary.csv"
    dict_df.to_csv(dict_file, index=False)
    logging.info(f"✓ Data dictionary saved to: {dict_file}")

if __name__ == "__main__":
    logging.info("Starting data cleaning pipeline...\n")
    
    # Load raw data
    df_raw = load_data()
    
    # Clean data
    df_clean = clean_data(df_raw)
    
    # Calculate percentiles
    df_final = calculate_percentiles(df_clean)
    
    # Generate summary stats
    generate_summary_stats(df_final)
    
    # Save cleaned data
    save_cleaned_data(df_final)
    
    logging.info("\n" + "="*60)
    logging.info("✓ DATA CLEANING COMPLETE!")
    logging.info("="*60)
    logging.info(f"Original records: {len(df_raw)}")
    logging.info(f"Final records: {len(df_final)}")
    logging.info(f"Records removed: {len(df_raw) - len(df_final)}")

