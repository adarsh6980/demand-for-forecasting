"""
Data Ingestion Module
Handles loading and validation of POS and external data.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pos_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate POS (Point of Sale) data from CSV.
    
    Args:
        filepath: Path to the POS CSV file
        
    Returns:
        DataFrame with columns: date, sku, units_sold, price, promo_flag
    """
    logger.info(f"Loading POS data from {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Validate required columns
    required_cols = ['date', 'sku', 'units_sold', 'price', 'promo_flag']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Data quality checks
    df = df.dropna(subset=['date', 'sku', 'units_sold'])
    df['units_sold'] = df['units_sold'].clip(lower=0)
    df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)
    
    logger.info(f"Loaded {len(df)} POS records for {df['sku'].nunique()} SKUs")
    return df


def load_external_data(filepath: str) -> pd.DataFrame:
    """
    Load external data (e.g., weather) from CSV.
    
    Args:
        filepath: Path to external data CSV
        
    Returns:
        DataFrame with date and external features
    """
    logger.info(f"Loading external data from {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Fill missing values with defaults
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    if 'precipitation' in df.columns:
        df['precipitation'] = df['precipitation'].fillna(0)
    
    logger.info(f"Loaded {len(df)} external data records")
    return df


def merge_data(pos_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge POS data with external data on date.
    
    Args:
        pos_df: POS DataFrame
        external_df: External data DataFrame
        
    Returns:
        Merged DataFrame
    """
    merged = pos_df.merge(external_df, on='date', how='left')
    
    # Fill any missing external data
    numeric_cols = merged.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())
    
    logger.info(f"Merged data: {len(merged)} records")
    return merged


def load_stock_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load current stock levels. Returns default if no file provided.
    
    Args:
        filepath: Optional path to stock CSV
        
    Returns:
        DataFrame with sku and current_stock columns
    """
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)
    
    # Default stock levels
    default_stock = {
        'sku': ['MILK_1L', 'BREAD_LOAF', 'EGGS_12', 'SODA_500ML', 'CHIPS_150G'],
        'current_stock': [50, 30, 25, 80, 60]
    }
    return pd.DataFrame(default_stock)


def get_sku_list(pos_df: pd.DataFrame) -> list:
    """Get unique list of SKUs from POS data."""
    return pos_df['sku'].unique().tolist()


if __name__ == "__main__":
    # Quick test
    base_path = Path(__file__).parent.parent / "data"
    pos = load_pos_data(base_path / "pos_data.csv")
    external = load_external_data(base_path / "external_data.csv")
    merged = merge_data(pos, external)
    print(merged.head())
