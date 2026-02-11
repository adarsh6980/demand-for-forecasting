"""
Zara Dataset Cleaner & Transformer
Cleans the Zara product catalog CSV and transforms it into
the daily time-series format expected by the demand forecasting pipeline.

Input:  data/zara.csv (252 products, single-day catalog scrape)
Output: data/pos_data.csv, data/external_data.csv, config/business_rules.yml
"""
import pandas as pd
import numpy as np
import yaml
import shutil
import requests
import time
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CONFIG_PATH = BASE_PATH / "config"

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & CLEAN RAW ZARA DATA
# ═══════════════════════════════════════════════════════════════════════

def load_and_clean_zara(filepath: str) -> pd.DataFrame:
    """
    Load and clean the raw Zara CSV file.
    
    Cleaning steps:
    1. Parse semicolon-delimited file
    2. Drop duplicates
    3. Handle missing names/descriptions
    4. Standardize SKU names
    5. Fix data types
    6. Remove invalid records
    """
    logger.info(f"Loading Zara data from {filepath}...")
    
    df = pd.read_csv(filepath, sep=';')
    logger.info(f"  Raw: {len(df)} rows, {len(df.columns)} columns")
    
    # --- Drop exact duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info(f"  Removed {before - len(df)} duplicate rows")
    
    # --- Drop duplicates by Product ID (keep first) ---
    before = len(df)
    df = df.drop_duplicates(subset=['Product ID'], keep='first')
    if len(df) < before:
        logger.info(f"  Removed {before - len(df)} duplicate Product IDs")
    
    # --- Handle missing product names ---
    missing_names = df['name'].isnull().sum()
    if missing_names > 0:
        # Fill from description or SKU
        df['name'] = df['name'].fillna(df['description'].str[:50])
        df['name'] = df['name'].fillna(df['sku'])
        logger.info(f"  Filled {missing_names} missing product names")
    
    # --- Handle missing descriptions ---
    missing_desc = df['description'].isnull().sum()
    if missing_desc > 0:
        df['description'] = df['description'].fillna('No description available')
        logger.info(f"  Filled {missing_desc} missing descriptions")
    
    # --- Create clean SKU name (human-readable) ---
    df['clean_sku'] = (
        df['name']
        .str.upper()
        .str.strip()
        .str.replace(r'[^A-Z0-9\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
        .str[:40]  # Truncate long names
    )
    
    # Handle duplicate SKU names by appending category
    sku_counts = df['clean_sku'].value_counts()
    duplicated_skus = sku_counts[sku_counts > 1].index
    for dup_sku in duplicated_skus:
        mask = df['clean_sku'] == dup_sku
        indices = df[mask].index
        for i, idx in enumerate(indices):
            if i > 0:
                section = df.loc[idx, 'section']
                df.loc[idx, 'clean_sku'] = f"{dup_sku}_{section}"
    
    # --- Validate & fix data types ---
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['Sales Volume'] = pd.to_numeric(df['Sales Volume'], errors='coerce')
    
    # Remove records with invalid price or sales
    before = len(df)
    df = df.dropna(subset=['price', 'Sales Volume'])
    df = df[df['price'] > 0]
    df = df[df['Sales Volume'] > 0]
    if len(df) < before:
        logger.info(f"  Removed {before - len(df)} records with invalid price/sales")
    
    # --- Standardize promotion & seasonal flags ---
    df['promo_flag'] = (df['Promotion'].str.strip().str.upper() == 'YES').astype(int)
    df['seasonal_flag'] = (df['Seasonal'].str.strip().str.upper() == 'YES').astype(int)
    
    # --- Parse scraped_at date ---
    df['scraped_date'] = pd.to_datetime(df['scraped_at'], errors='coerce').dt.date
    
    # --- Log cleaning summary ---
    logger.info(f"\n  ✅ Cleaned Zara data: {len(df)} products")
    logger.info(f"  Categories: {df['terms'].value_counts().to_dict()}")
    logger.info(f"  Sections: {df['section'].value_counts().to_dict()}")
    logger.info(f"  Promotions: {df['promo_flag'].sum()} products on promotion")
    logger.info(f"  Seasonal: {df['seasonal_flag'].sum()} seasonal products")
    logger.info(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    logger.info(f"  Sales range: {df['Sales Volume'].min()} - {df['Sales Volume'].max()}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: EXPAND TO DAILY TIME-SERIES
# ═══════════════════════════════════════════════════════════════════════

def expand_to_daily_timeseries(df: pd.DataFrame, num_days: int = 180) -> pd.DataFrame:
    """
    Expand the single-snapshot catalog into a daily time-series.
    
    The Zara CSV has aggregate Sales Volume per product. We distribute this
    across a realistic daily pattern using:
    - Weekly seasonality (weekends higher)
    - Monthly trends (seasonal items peak in season)
    - Promotion effects (promoted items get sales bumps)
    - Random noise for realism
    - Category-specific patterns (jackets peak in winter, t-shirts in summer)
    
    This creates realistic daily data that the forecasting model can learn from.
    """
    logger.info(f"Expanding to {num_days}-day time series for {len(df)} products...")
    
    # Select top 20 products by sales volume for manageable dataset
    top_products = df.nlargest(20, 'Sales Volume').copy()
    logger.info(f"  Selected top 20 products by sales volume")
    
    # Date range: 6 months ending at scrape date
    end_date = pd.Timestamp('2024-02-19')  # scrape date
    start_date = end_date - pd.Timedelta(days=num_days - 1)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    np.random.seed(42)
    all_records = []
    
    for _, product in top_products.iterrows():
        sku = product['clean_sku']
        total_sales = product['Sales Volume']
        base_price = product['price']
        is_promo = product['promo_flag']
        is_seasonal = product['seasonal_flag']
        category = product['terms']
        
        # Calculate average daily demand
        avg_daily = total_sales / num_days
        
        for date in dates:
            day_of_week = date.dayofweek
            month = date.month
            day_of_year = date.dayofyear
            
            # --- Base demand ---
            daily_demand = avg_daily
            
            # --- Weekly pattern (weekends +30%) ---
            if day_of_week >= 5:  # Saturday, Sunday
                daily_demand *= 1.3
            elif day_of_week == 4:  # Friday
                daily_demand *= 1.15
            elif day_of_week == 0:  # Monday
                daily_demand *= 0.85
            
            # --- Category-specific seasonal patterns ---
            if category == 'jackets':
                # Peak in autumn/winter (Oct-Feb)
                seasonal_factor = 1.0 + 0.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
            elif category == 't-shirts':
                # Peak in spring/summer (Apr-Aug)
                seasonal_factor = 1.0 + 0.4 * np.cos(2 * np.pi * (day_of_year - 196) / 365)
            elif category == 'sweaters':
                # Peak in autumn/winter
                seasonal_factor = 1.0 + 0.45 * np.cos(2 * np.pi * (day_of_year - 1) / 365)
            elif category == 'shoes':
                # Mild seasonality, peaks in spring and autumn
                seasonal_factor = 1.0 + 0.2 * np.cos(4 * np.pi * day_of_year / 365)
            elif category == 'jeans':
                # Relatively stable with slight back-to-school bump
                seasonal_factor = 1.0 + 0.15 * np.cos(2 * np.pi * (day_of_year - 240) / 365)
            else:
                seasonal_factor = 1.0
            
            daily_demand *= seasonal_factor
            
            # --- Promotion effect (+40-80% during promo periods) ---
            promo_today = 0
            if is_promo:
                # Assume promo runs for ~2-3 weeks around certain periods
                if month in [1, 6, 7, 11, 12]:  # Sale seasons (Jan sales, Summer sales, Black Friday)
                    promo_today = 1
                    daily_demand *= np.random.uniform(1.4, 1.8)
            
            # --- Seasonal item boost ---
            if is_seasonal:
                if category in ['jackets', 'sweaters'] and month in [10, 11, 12, 1, 2]:
                    daily_demand *= 1.25
                elif category in ['t-shirts'] and month in [5, 6, 7, 8]:
                    daily_demand *= 1.25
            
            # --- Holiday spikes ---
            # Black Friday (late Nov), Christmas, New Year sales
            if month == 11 and 22 <= date.day <= 28:
                daily_demand *= np.random.uniform(2.0, 3.0)
                promo_today = 1
            elif month == 12 and 15 <= date.day <= 24:
                daily_demand *= np.random.uniform(1.5, 2.5)
            elif month == 1 and date.day <= 7:
                daily_demand *= np.random.uniform(1.3, 2.0)
                promo_today = 1
            
            # --- Random noise (±20%) ---
            noise = np.random.normal(1.0, 0.2)
            daily_demand *= max(0.3, noise)
            
            # --- Price variation (±5% normal, -10-25% during promos) ---
            if promo_today:
                price_today = base_price * np.random.uniform(0.75, 0.90)
            else:
                price_today = base_price * np.random.uniform(0.95, 1.05)
            
            all_records.append({
                'date': date,
                'sku': sku,
                'units_sold': max(0, int(round(daily_demand))),
                'price': round(price_today, 2),
                'promo_flag': promo_today
            })
    
    pos_df = pd.DataFrame(all_records)
    pos_df = pos_df.sort_values(['sku', 'date']).reset_index(drop=True)
    
    logger.info(f"  ✅ Generated {len(pos_df)} daily records for {pos_df['sku'].nunique()} SKUs")
    logger.info(f"  Date range: {pos_df['date'].min().date()} to {pos_df['date'].max().date()}")
    
    return pos_df


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: FETCH REAL WEATHER DATA
# ═══════════════════════════════════════════════════════════════════════

def fetch_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real weather from Open-Meteo API.
    Using New York City coordinates (Zara US store context).
    """
    logger.info(f"Fetching real weather data ({start_date} to {end_date})...")
    
    all_weather = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current < end:
        chunk_end = min(current + pd.DateOffset(years=1) - pd.DateOffset(days=1), end)
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=40.7128&longitude=-74.0060"  # NYC
            f"&start_date={current.strftime('%Y-%m-%d')}"
            f"&end_date={chunk_end.strftime('%Y-%m-%d')}"
            f"&daily=temperature_2m_mean,precipitation_sum"
            f"&timezone=America%2FNew_York"
        )
        
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if 'daily' in data:
                chunk_df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'temperature': data['daily']['temperature_2m_mean'],
                    'precipitation': data['daily']['precipitation_sum']
                })
                all_weather.append(chunk_df)
                logger.info(f"  Weather fetched: {current.date()} to {chunk_end.date()}")
        except Exception as e:
            logger.warning(f"  Weather API failed: {e}")
        
        current = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.5)
    
    if not all_weather:
        logger.warning("  API failed, generating fallback weather data")
        dates = pd.date_range(start_date, end_date)
        np.random.seed(42)
        day_of_year = dates.dayofyear
        temp = 12 + 14 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 3, len(dates))
        precip = np.random.exponential(2, len(dates))
        precip[np.random.random(len(dates)) > 0.4] = 0
        weather_df = pd.DataFrame({'date': dates, 'temperature': temp.round(1), 'precipitation': precip.round(1)})
    else:
        weather_df = pd.concat(all_weather, ignore_index=True)
    
    weather_df['temperature'] = weather_df['temperature'].ffill().bfill()
    weather_df['precipitation'] = weather_df['precipitation'].fillna(0)
    
    weather_df['weather_condition'] = 'sunny'
    weather_df.loc[weather_df['precipitation'] > 0.5, 'weather_condition'] = 'cloudy'
    weather_df.loc[weather_df['precipitation'] > 5.0, 'weather_condition'] = 'rainy'
    
    logger.info(f"  ✅ Weather data: {len(weather_df)} days")
    return weather_df


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE BUSINESS RULES
# ═══════════════════════════════════════════════════════════════════════

def generate_zara_business_rules(pos_df: pd.DataFrame, zara_df: pd.DataFrame) -> dict:
    """Generate business rules YAML from Zara data statistics."""
    logger.info("Generating Zara business rules...")
    
    skus_config = []
    for sku in pos_df['sku'].unique():
        sku_data = pos_df[pos_df['sku'] == sku]
        avg_daily = sku_data['units_sold'].mean()
        max_daily = sku_data['units_sold'].max()
        avg_price = sku_data['price'].mean()
        
        # Match back to Zara catalog for extra info
        zara_match = zara_df[zara_df['clean_sku'] == sku]
        category = zara_match['terms'].values[0] if len(zara_match) > 0 else 'clothing'
        
        # Category-specific perishability (fashion lifecycle)
        if category == 't-shirts':
            perishability = 60  # Basics rotate faster
        elif category in ['jackets', 'sweaters']:
            perishability = 120  # Outerwear stays longer
        elif category == 'jeans':
            perishability = 150  # Jeans have longer shelf life
        else:
            perishability = 90
        
        config = {
            'sku': sku,
            'max_shelf_capacity': int(max(max_daily * 2, avg_daily * 5, 30)),
            'unit_cost': round(float(avg_price * 0.4), 2),  # Zara ~40% COGS
            'max_budget_per_order': int(max(1000, avg_daily * avg_price * 7)),
            'perishability_days': perishability,
            'safety_stock_days': 3
        }
        skus_config.append(config)
    
    return {'skus': skus_config}


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════

def backup_existing():
    """Backup existing data files."""
    for name in ['pos_data.csv', 'external_data.csv']:
        src = DATA_PATH / name
        if src.exists():
            backup = DATA_PATH / name.replace('.csv', '_backup.csv')
            shutil.copy2(src, backup)
            logger.info(f"  Backed up {name}")


def print_data_report(pos_df: pd.DataFrame, weather_df: pd.DataFrame):
    """Print a summary report of the generated data."""
    print("\n" + "=" * 60)
    print("  🛍️  ZARA DATASET — CLEANING & INTEGRATION REPORT")
    print("=" * 60)
    
    print(f"\n  📊 POS Data (pos_data.csv)")
    print(f"     Records:    {len(pos_df):,}")
    print(f"     SKUs:       {pos_df['sku'].nunique()}")
    print(f"     Date range: {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
    print(f"     Avg daily sales: {pos_df['units_sold'].mean():.0f} units")
    print(f"     Price range: ${pos_df['price'].min():.2f} - ${pos_df['price'].max():.2f}")
    print(f"     Promo days:  {pos_df['promo_flag'].sum():,} ({pos_df['promo_flag'].mean()*100:.1f}%)")
    
    print(f"\n  🌤️  Weather Data (external_data.csv)")
    print(f"     Days: {len(weather_df)}")
    print(f"     Avg temp: {weather_df['temperature'].mean():.1f}°C")
    print(f"     Rainy days: {(weather_df['weather_condition'] == 'rainy').sum()}")
    
    print(f"\n  📦 Top 5 Products by Sales Volume:")
    top5 = pos_df.groupby('sku')['units_sold'].sum().nlargest(5)
    for sku, total in top5.items():
        print(f"     {sku}: {total:,} units")
    
    print("\n" + "=" * 60)
    print("  ✅ Ready! Launch dashboard with:")
    print("     streamlit run src/dashboard.py")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🛍️  Zara Dataset Cleaner & Integrator")
    print("=" * 60)
    
    zara_file = DATA_PATH / "zara.csv"
    if not zara_file.exists():
        logger.error(f"❌ Could not find {zara_file}")
        sys.exit(1)
    
    # Step 1: Clean raw data
    print("\n📋 Step 1: Cleaning raw Zara data...")
    zara_df = load_and_clean_zara(str(zara_file))
    
    # Step 2: Expand to daily time-series
    print("\n📈 Step 2: Generating daily time-series (180 days)...")
    pos_df = expand_to_daily_timeseries(zara_df, num_days=180)
    
    # Step 3: Fetch weather
    print("\n🌤️  Step 3: Fetching real weather data...")
    start = pos_df['date'].min().strftime('%Y-%m-%d')
    end = pos_df['date'].max().strftime('%Y-%m-%d')
    weather_df = fetch_weather_data(start, end)
    
    # Step 4: Backup & save
    print("\n💾 Step 4: Saving files...")
    backup_existing()
    
    pos_df.to_csv(DATA_PATH / 'pos_data.csv', index=False)
    logger.info(f"  Saved pos_data.csv ({len(pos_df)} rows)")
    
    weather_df.to_csv(DATA_PATH / 'external_data.csv', index=False)
    logger.info(f"  Saved external_data.csv ({len(weather_df)} rows)")
    
    # Step 5: Business rules
    print("\n⚙️  Step 5: Generating business rules...")
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    rules = generate_zara_business_rules(pos_df, zara_df)
    with open(CONFIG_PATH / 'business_rules.yml', 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)
    logger.info(f"  Saved business_rules.yml for {len(rules['skus'])} SKUs")
    
    # Report
    print_data_report(pos_df, weather_df)


if __name__ == "__main__":
    main()
