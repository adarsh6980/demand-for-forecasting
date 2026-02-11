"""
Real-World Retail Dataset Downloader & Transformer
Downloads real retail sales data and transforms it into the format expected
by the demand forecasting pipeline.

Sources (in order of preference):
1. Kaggle "Store Item Demand Forecasting" dataset (requires kaggle CLI)
2. UCI Online Retail II dataset (direct download, no auth)

Also fetches real historical weather data from Open-Meteo API (free, no key).
"""
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import shutil
import yaml
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CONFIG_PATH = BASE_PATH / "config"


# ═══════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def try_kaggle_download() -> pd.DataFrame:
    """
    Try downloading the Kaggle Store Item Demand Forecasting dataset.
    Requires: pip install kaggle + ~/.kaggle/kaggle.json
    Dataset: 5 years of daily sales for 50 items across 10 stores.
    """
    logger.info("Attempting Kaggle download...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "c/demand-forecasting-kernels-only",
             "-p", str(DATA_PATH / "raw"), "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle CLI failed: {result.stderr}")

        train_file = DATA_PATH / "raw" / "train.csv"
        if not train_file.exists():
            raise FileNotFoundError("Kaggle download succeeded but train.csv not found")

        df = pd.read_csv(train_file, parse_dates=['date'])
        logger.info(f"✅ Kaggle dataset loaded: {len(df)} rows")
        return df

    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Kaggle download failed: {e}")
        return None


def download_uci_online_retail() -> pd.DataFrame:
    """
    Download UCI Online Retail II dataset (real UK retailer, ~1M transactions).
    Direct download from UCI ML Repository - no authentication needed.
    """
    logger.info("Downloading UCI Online Retail II dataset...")

    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunks = []
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Downloading: {pct:.0f}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)
        print()

        content = b''.join(chunks)
        logger.info(f"  Downloaded {len(content) // 1024 // 1024}MB")

        # Extract the zip
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_list = z.namelist()
            logger.info(f"  Archive contains: {file_list}")

            # Find the Excel file
            xlsx_files = [f for f in file_list if f.endswith('.xlsx')]
            if not xlsx_files:
                raise FileNotFoundError("No xlsx file in archive")

            with z.open(xlsx_files[0]) as excel_file:
                logger.info(f"  Reading {xlsx_files[0]}...")
                df = pd.read_excel(
                    io.BytesIO(excel_file.read()),
                    engine='openpyxl'
                )

        logger.info(f"✅ UCI Online Retail II loaded: {len(df)} rows, columns: {list(df.columns)}")
        return df

    except Exception as e:
        logger.error(f"UCI download failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# DATA TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════

def transform_kaggle_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform Kaggle store-item data into pos_data format.
    Kaggle columns: date, store, item, sales
    """
    logger.info("Transforming Kaggle data...")

    # Create readable SKU names from store + item
    item_names = {
        1: "WIRELESS_HEADPHONES", 2: "USB_CHARGING_CABLE", 3: "PHONE_CASE_CLEAR",
        4: "BLUETOOTH_SPEAKER", 5: "SCREEN_PROTECTOR", 6: "CAR_PHONE_MOUNT",
        7: "LAPTOP_SLEEVE_15", 8: "WIRELESS_MOUSE", 9: "LED_DESK_LAMP",
        10: "PORTABLE_CHARGER", 11: "WEBCAM_HD_1080P", 12: "KEYBOARD_WIRELESS",
        13: "HDMI_CABLE_2M", 14: "USB_HUB_4PORT", 15: "EARBUDS_WIRED",
        16: "TABLET_STAND", 17: "POWER_STRIP_6OUT", 18: "ETHERNET_CABLE_5M",
        19: "FLASH_DRIVE_64GB", 20: "PHONE_RING_HOLDER",
        21: "COTTON_T_SHIRT", 22: "DENIM_JEANS_SLIM", 23: "CANVAS_SNEAKERS",
        24: "LEATHER_BELT_BLACK", 25: "WOOL_BEANIE_HAT",
        26: "SUNGLASSES_AVIATOR", 27: "BACKPACK_DAILY", 28: "WATER_BOTTLE_750ML",
        29: "YOGA_MAT_STANDARD", 30: "RESISTANCE_BANDS_SET",
        31: "SCENTED_CANDLE_SOY", 32: "CERAMIC_MUG_350ML", 33: "THROW_PILLOW_COVER",
        34: "BATH_TOWEL_SET", 35: "KITCHEN_TIMER_DIGITAL",
        36: "NOTEBOOK_A5_LINED", 37: "BALLPOINT_PEN_PACK", 38: "STICKY_NOTES_NEON",
        39: "DESK_ORGANIZER", 40: "FILING_FOLDERS_10PK",
        41: "DOG_TREAT_BISCUITS", 42: "CAT_TOY_FEATHER", 43: "PET_BOWL_STEEL",
        44: "PLANT_POT_CERAMIC", 45: "GARDEN_GLOVES_PAIR",
        46: "FIRST_AID_KIT", 47: "REUSABLE_BAGS_5PK", 48: "LUNCH_BOX_INSULATED",
        49: "UMBRELLA_COMPACT", 50: "KEYCHAIN_LEATHER"
    }

    # Pick top 5 stores to keep dataset manageable (like 5 retail chains)
    store_names = {1: "DOWNTOWN", 2: "MALL", 3: "SUBURB", 4: "AIRPORT", 5: "OUTLET"}
    selected_stores = [1, 2, 3, 4, 5]

    # Select top 20 items for a focused analysis
    selected_items = list(range(1, 21))

    df = raw_df[
        raw_df['store'].isin(selected_stores) & raw_df['item'].isin(selected_items)
    ].copy()

    # Map to readable names
    df['sku'] = df.apply(
        lambda r: f"{store_names.get(r['store'], f'STORE_{r.store}')}_{item_names.get(r['item'], f'ITEM_{r.item}')}",
        axis=1
    )

    # Generate realistic prices based on item (dataset doesn't include prices)
    np.random.seed(42)
    base_prices = {item: round(np.random.uniform(3.99, 49.99), 2) for item in selected_items}

    df['price'] = df['item'].map(base_prices)
    # Add small price variation (±10%)
    df['price'] = df['price'] * (1 + np.random.uniform(-0.1, 0.1, len(df)))
    df['price'] = df['price'].round(2)

    # Detect promotions: sales > 1.5x the 14-day rolling average
    df = df.sort_values(['store', 'item', 'date'])
    df['rolling_avg'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df['promo_flag'] = (df['sales'] > df['rolling_avg'] * 1.5).astype(int)

    # Build final pos_data format
    pos_df = df[['date', 'sku', 'sales', 'price', 'promo_flag']].copy()
    pos_df = pos_df.rename(columns={'sales': 'units_sold'})

    logger.info(f"  Transformed: {len(pos_df)} rows, {pos_df['sku'].nunique()} SKUs")
    return pos_df


def transform_uci_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform UCI Online Retail II data into pos_data format.
    UCI columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
    """
    logger.info("Transforming UCI Online Retail II data...")

    # Standardize column names (UCI has varying names across versions)
    col_map = {}
    for col in raw_df.columns:
        cl = col.lower().strip()
        if 'invoice' in cl and 'date' in cl:
            col_map[col] = 'invoicedate'
        elif 'invoice' in cl:
            col_map[col] = 'invoice'
        elif 'stock' in cl:
            col_map[col] = 'stockcode'
        elif 'description' in cl:
            col_map[col] = 'description'
        elif 'quantity' in cl:
            col_map[col] = 'quantity'
        elif 'price' in cl:
            col_map[col] = 'price'
        elif 'customer' in cl:
            col_map[col] = 'customer_id'
        elif 'country' in cl:
            col_map[col] = 'country'
    raw_df = raw_df.rename(columns=col_map)

    logger.info(f"  Columns after rename: {list(raw_df.columns)}")

    # Clean data
    df = raw_df.copy()
    df = df.dropna(subset=['description', 'quantity', 'price'])
    df = df[df['quantity'] > 0]  # Remove returns/cancellations
    df = df[df['price'] > 0]  # Remove free items

    # Parse dates
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df = df.dropna(subset=['invoicedate'])
    df['date'] = df['invoicedate'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Clean description to create SKU names
    df['description'] = df['description'].astype(str).str.strip()
    df['sku'] = df['description'].str.upper().str.replace(r'[^A-Z0-9\s]', '', regex=True)
    df['sku'] = df['sku'].str.replace(r'\s+', '_', regex=True)
    df['sku'] = df['sku'].str[:50]  # Truncate long names

    # Focus on UK (largest market in this dataset) for consistent patterns
    if 'country' in df.columns:
        uk_df = df[df['country'].str.strip() == 'United Kingdom']
        if len(uk_df) > 10000:
            df = uk_df
            logger.info(f"  Filtered to UK: {len(df)} rows")

    # Aggregate to daily level per SKU
    daily = df.groupby(['date', 'sku']).agg(
        units_sold=('quantity', 'sum'),
        price=('price', 'mean')
    ).reset_index()

    # Keep top 20 SKUs by total sales volume for a focused dataset
    top_skus = daily.groupby('sku')['units_sold'].sum().nlargest(20).index.tolist()
    daily = daily[daily['sku'].isin(top_skus)]

    # Ensure continuous date range per SKU (fill gaps with 0 sales)
    date_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    all_combos = pd.MultiIndex.from_product([date_range, top_skus], names=['date', 'sku'])
    daily = daily.set_index(['date', 'sku']).reindex(all_combos).reset_index()

    # Forward-fill prices, fill missing sales with 0
    daily['price'] = daily.groupby('sku')['price'].transform(lambda x: x.ffill().bfill())
    daily['units_sold'] = daily['units_sold'].fillna(0).astype(int)
    daily['price'] = daily['price'].round(2)

    # Detect promotions: sales > 1.5x the 14-day rolling average
    daily = daily.sort_values(['sku', 'date'])
    daily['rolling_avg'] = daily.groupby('sku')['units_sold'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    daily['promo_flag'] = ((daily['units_sold'] > daily['rolling_avg'] * 1.5) &
                           (daily['units_sold'] > 0)).astype(int)

    pos_df = daily[['date', 'sku', 'units_sold', 'price', 'promo_flag']].copy()

    logger.info(f"  Transformed: {len(pos_df)} rows, {pos_df['sku'].nunique()} SKUs")
    logger.info(f"  Date range: {pos_df['date'].min()} to {pos_df['date'].max()}")
    logger.info(f"  SKUs: {pos_df['sku'].unique().tolist()}")
    return pos_df


# ═══════════════════════════════════════════════════════════════════════
# WEATHER DATA (Open-Meteo API - free, no key)
# ═══════════════════════════════════════════════════════════════════════

def fetch_weather_data(start_date: str, end_date: str,
                       latitude: float = 51.5074,  # London (for UCI UK data)
                       longitude: float = -0.1278) -> pd.DataFrame:
    """
    Fetch real historical weather data from Open-Meteo API.
    Free API, no key needed. Rate limit: be polite.
    """
    logger.info(f"Fetching weather data from Open-Meteo ({start_date} to {end_date})...")

    # Open-Meteo has a max range limit, so we fetch in yearly chunks
    all_weather = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    current_start = start
    while current_start < end:
        current_end = min(current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end)

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={current_start.strftime('%Y-%m-%d')}"
            f"&end_date={current_end.strftime('%Y-%m-%d')}"
            f"&daily=temperature_2m_mean,precipitation_sum"
            f"&timezone=auto"
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
                logger.info(f"  Fetched weather: {current_start.strftime('%Y-%m-%d')} "
                          f"to {current_end.strftime('%Y-%m-%d')} ({len(chunk_df)} days)")
            else:
                logger.warning(f"  No daily data in response for {current_start.date()}")

        except Exception as e:
            logger.warning(f"  Weather API call failed for {current_start.date()}: {e}")

        current_start = current_end + pd.DateOffset(days=1)
        time.sleep(0.5)  # Be polite to the free API

    if not all_weather:
        logger.warning("  Weather API failed entirely. Generating synthetic weather.")
        return generate_fallback_weather(start_date, end_date)

    weather_df = pd.concat(all_weather, ignore_index=True)

    # Add weather condition label
    weather_df['weather_condition'] = 'sunny'
    weather_df.loc[weather_df['precipitation'] > 0.5, 'weather_condition'] = 'cloudy'
    weather_df.loc[weather_df['precipitation'] > 5.0, 'weather_condition'] = 'rainy'

    # Fill any NaN
    weather_df['temperature'] = weather_df['temperature'].ffill().bfill()
    weather_df['precipitation'] = weather_df['precipitation'].fillna(0)

    logger.info(f"✅ Weather data ready: {len(weather_df)} days")
    return weather_df


def generate_fallback_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic weather data as fallback if API fails."""
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(42)

    # Seasonal temperature pattern (Northern Hemisphere)
    day_of_year = dates.dayofyear
    temp = 10 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 3, len(dates))

    # Precipitation
    precip = np.random.exponential(2, len(dates))
    precip[np.random.random(len(dates)) > 0.4] = 0  # 60% dry days

    weather_df = pd.DataFrame({
        'date': dates,
        'temperature': temp.round(1),
        'precipitation': precip.round(1)
    })

    weather_df['weather_condition'] = 'sunny'
    weather_df.loc[weather_df['precipitation'] > 0.5, 'weather_condition'] = 'cloudy'
    weather_df.loc[weather_df['precipitation'] > 5.0, 'weather_condition'] = 'rainy'

    return weather_df


# ═══════════════════════════════════════════════════════════════════════
# BUSINESS RULES GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_business_rules(pos_df: pd.DataFrame) -> dict:
    """Auto-generate business rules YAML based on actual data statistics."""
    logger.info("Generating business rules from data...")

    skus_config = []
    for sku in pos_df['sku'].unique():
        sku_data = pos_df[pos_df['sku'] == sku]
        avg_daily = sku_data['units_sold'].mean()
        max_daily = sku_data['units_sold'].max()
        avg_price = sku_data['price'].mean()

        config = {
            'sku': sku,
            'max_shelf_capacity': int(max(max_daily * 2, avg_daily * 5, 50)),
            'unit_cost': round(float(avg_price * 0.6), 2),  # ~60% margin
            'max_budget_per_order': int(max(500, avg_daily * avg_price * 7)),
            'perishability_days': 90,
            'safety_stock_days': 3
        }
        skus_config.append(config)

    rules = {'skus': skus_config}
    logger.info(f"  Generated rules for {len(skus_config)} SKUs")
    return rules


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def backup_existing_data():
    """Backup existing data files before overwriting."""
    for filename in ['pos_data.csv', 'external_data.csv']:
        src = DATA_PATH / filename
        if src.exists():
            backup = DATA_PATH / filename.replace('.csv', '_backup.csv')
            shutil.copy2(src, backup)
            logger.info(f"  Backed up {filename} → {backup.name}")


def main():
    """Main entry point: download, transform, save."""
    print("=" * 60)
    print("  Real-World Retail Dataset Downloader")
    print("=" * 60)

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)

    # Step 1: Try to download real data
    pos_df = None
    source = None

    # Strategy 1: Kaggle
    raw_kaggle = try_kaggle_download()
    if raw_kaggle is not None:
        pos_df = transform_kaggle_data(raw_kaggle)
        source = "Kaggle Store Item Demand Forecasting"

    # Strategy 2: UCI Online Retail II
    if pos_df is None:
        raw_uci = download_uci_online_retail()
        if raw_uci is not None:
            pos_df = transform_uci_data(raw_uci)
            source = "UCI Online Retail II (Real UK Retailer)"

    if pos_df is None:
        logger.error("❌ Failed to download any dataset. Please check your internet connection.")
        sys.exit(1)

    # Step 2: Fetch weather data
    start_date = pos_df['date'].min().strftime('%Y-%m-%d')
    end_date = pos_df['date'].max().strftime('%Y-%m-%d')

    # Use London coords for UCI (UK retailer) or NYC for Kaggle
    if source and "UCI" in source:
        weather_df = fetch_weather_data(start_date, end_date, latitude=51.5074, longitude=-0.1278)
    else:
        weather_df = fetch_weather_data(start_date, end_date, latitude=40.7128, longitude=-74.0060)

    # Step 3: Backup & save
    print("\n📁 Saving data files...")
    backup_existing_data()

    pos_df.to_csv(DATA_PATH / 'pos_data.csv', index=False)
    logger.info(f"  Saved pos_data.csv ({len(pos_df)} rows)")

    weather_df.to_csv(DATA_PATH / 'external_data.csv', index=False)
    logger.info(f"  Saved external_data.csv ({len(weather_df)} rows)")

    # Step 4: Generate business rules
    rules = generate_business_rules(pos_df)
    with open(CONFIG_PATH / 'business_rules.yml', 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)
    logger.info(f"  Saved business_rules.yml")

    # Step 5: Clean up raw downloads
    raw_dir = DATA_PATH / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Summary
    print("\n" + "=" * 60)
    print(f"  ✅ Real-world dataset ready!")
    print(f"  Source: {source}")
    print(f"  POS records: {len(pos_df):,}")
    print(f"  SKUs: {pos_df['sku'].nunique()}")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  Weather days: {len(weather_df)}")
    print("=" * 60)
    print(f"\n  Files saved to: {DATA_PATH}")
    print(f"    • pos_data.csv")
    print(f"    • external_data.csv")
    print(f"    • config/business_rules.yml")
    print(f"\n  Next steps:")
    print(f"    1. python src/data_ingestion.py   (verify loading)")
    print(f"    2. python src/forecasting.py      (train models)")
    print(f"    3. streamlit run src/dashboard.py  (launch dashboard)")


if __name__ == "__main__":
    main()
