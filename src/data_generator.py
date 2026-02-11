"""
Realistic Data Generator
Generates synthetic but realistic retail demand data with:
- Seasonality (weekly, monthly, yearly patterns)
- Trend components
- Promotional effects
- Weather impact
- Random noise
- Concept drift (for testing drift detection)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import random

np.random.seed(42)


class RealisticDataGenerator:
    """Generates realistic retail demand data with various patterns."""
    
    # SKU configurations with realistic parameters
    SKU_CONFIGS = {
        'MILK_1L': {
            'base_demand': 80,
            'price': 2.99,
            'perishability_days': 7,
            'weekly_pattern': [0.9, 0.85, 0.8, 0.85, 1.0, 1.3, 1.2],  # Mon-Sun
            'promo_lift': 1.4,
            'weather_sensitivity': 0.1,  # Less affected by weather
            'trend': 0.0005,  # Slight growth
        },
        'BREAD_LOAF': {
            'base_demand': 60,
            'price': 3.49,
            'perishability_days': 3,
            'weekly_pattern': [0.95, 0.9, 0.85, 0.9, 1.0, 1.2, 1.15],
            'promo_lift': 1.3,
            'weather_sensitivity': 0.05,
            'trend': 0.0003,
        },
        'EGGS_12': {
            'base_demand': 45,
            'price': 4.99,
            'perishability_days': 14,
            'weekly_pattern': [0.8, 0.75, 0.75, 0.85, 1.0, 1.4, 1.3],
            'promo_lift': 1.25,
            'weather_sensitivity': 0.02,
            'trend': 0.0004,
        },
        'SODA_500ML': {
            'base_demand': 120,
            'price': 1.99,
            'perishability_days': 180,
            'weekly_pattern': [0.85, 0.8, 0.8, 0.85, 1.1, 1.3, 1.25],
            'promo_lift': 1.5,
            'weather_sensitivity': 0.3,  # Hot weather increases sales
            'trend': 0.0002,
        },
        'CHIPS_150G': {
            'base_demand': 90,
            'price': 2.49,
            'perishability_days': 90,
            'weekly_pattern': [0.8, 0.75, 0.8, 0.9, 1.15, 1.35, 1.2],
            'promo_lift': 1.45,
            'weather_sensitivity': 0.15,
            'trend': 0.0006,
        },
        'WATER_1L': {
            'base_demand': 150,
            'price': 0.99,
            'perishability_days': 365,
            'weekly_pattern': [0.9, 0.85, 0.85, 0.9, 1.05, 1.2, 1.2],
            'promo_lift': 1.2,
            'weather_sensitivity': 0.4,  # Very weather sensitive
            'trend': 0.0008,
        },
        'YOGURT_4PK': {
            'base_demand': 55,
            'price': 3.99,
            'perishability_days': 21,
            'weekly_pattern': [0.85, 0.8, 0.8, 0.85, 1.0, 1.3, 1.35],
            'promo_lift': 1.35,
            'weather_sensitivity': -0.1,  # Cold weather slightly increases
            'trend': 0.0007,
        },
        'COFFEE_250G': {
            'base_demand': 35,
            'price': 7.99,
            'perishability_days': 180,
            'weekly_pattern': [1.1, 1.05, 0.95, 0.95, 1.0, 0.95, 0.95],
            'promo_lift': 1.3,
            'weather_sensitivity': -0.15,  # Cold weather increases
            'trend': 0.0005,
        },
    }
    
    def __init__(self, start_date: str = '2024-01-01', num_days: int = 365):
        self.start_date = pd.to_datetime(start_date)
        self.num_days = num_days
        self.dates = pd.date_range(start=self.start_date, periods=num_days, freq='D')
        
    def _generate_weather(self) -> pd.DataFrame:
        """Generate realistic weather data with seasonal patterns."""
        weather_data = []
        
        for i, date in enumerate(self.dates):
            # Base temperature by month (Northern hemisphere pattern)
            month = date.month
            base_temps = {1: 5, 2: 7, 3: 12, 4: 16, 5: 21, 6: 26, 
                         7: 29, 8: 28, 9: 24, 10: 18, 11: 11, 12: 6}
            base_temp = base_temps[month]
            
            # Add daily variation
            temp = base_temp + np.random.normal(0, 4)
            temp = max(-5, min(40, temp))  # Clip to realistic range
            
            # Precipitation (more likely in certain months)
            precip_prob = 0.25 + 0.1 * np.sin(month * np.pi / 6)
            precipitation = np.random.exponential(5) if random.random() < precip_prob else 0
            
            # Weather condition
            if precipitation > 10:
                condition = 'rainy'
            elif precipitation > 0:
                condition = 'cloudy'
            else:
                condition = 'sunny'
            
            weather_data.append({
                'date': date,
                'temperature': round(temp, 1),
                'precipitation': round(precipitation, 1),
                'weather_condition': condition
            })
        
        return pd.DataFrame(weather_data)
    
    def _generate_promotions(self, sku: str) -> np.ndarray:
        """Generate realistic promotion patterns."""
        promos = np.zeros(self.num_days)
        
        # Weekly promotions (some SKUs have regular weekly deals)
        if random.random() > 0.5:
            promo_day = random.randint(4, 6)  # Fri-Sun more common
            for i in range(self.num_days):
                if self.dates[i].dayofweek == promo_day and random.random() > 0.6:
                    promos[i] = 1
        
        # Monthly big sales
        for i in range(self.num_days):
            if self.dates[i].day in [1, 15] and random.random() > 0.7:
                promos[i] = 1
                if i + 1 < self.num_days:
                    promos[i + 1] = 1
        
        # Holiday promotions
        holidays = [(1, 1), (2, 14), (7, 4), (10, 31), (11, 25), (12, 25)]
        for i in range(self.num_days):
            if (self.dates[i].month, self.dates[i].day) in holidays:
                for j in range(max(0, i-2), min(self.num_days, i+3)):
                    promos[j] = 1
        
        return promos
    
    def _add_drift(self, demand: np.ndarray, drift_start_pct: float = 0.7,
                   drift_type: str = 'gradual') -> np.ndarray:
        """Add concept drift to simulate changing patterns."""
        drift_start = int(self.num_days * drift_start_pct)
        
        if drift_type == 'gradual':
            # Gradual increase in demand
            for i in range(drift_start, self.num_days):
                progress = (i - drift_start) / (self.num_days - drift_start)
                demand[i] *= (1 + 0.3 * progress)  # Up to 30% increase
                
        elif drift_type == 'sudden':
            # Sudden shift
            demand[drift_start:] *= 1.25
            
        elif drift_type == 'seasonal_shift':
            # Weekly pattern shifts
            for i in range(drift_start, self.num_days):
                if self.dates[i].dayofweek in [0, 1, 2]:  # Mon-Wed become busier
                    demand[i] *= 1.2
                    
        return demand
    
    def generate_sku_data(self, sku: str, weather_df: pd.DataFrame,
                          add_drift: bool = True) -> pd.DataFrame:
        """Generate demand data for a single SKU."""
        config = self.SKU_CONFIGS.get(sku, self.SKU_CONFIGS['MILK_1L'])
        
        demand = np.zeros(self.num_days)
        promos = self._generate_promotions(sku)
        
        for i in range(self.num_days):
            date = self.dates[i]
            
            # Base demand
            base = config['base_demand']
            
            # Weekly pattern
            weekly_mult = config['weekly_pattern'][date.dayofweek]
            
            # Monthly seasonality (slight)
            monthly_mult = 1 + 0.1 * np.sin(date.month * np.pi / 6)
            
            # Trend
            trend_mult = 1 + config['trend'] * i
            
            # Weather effect
            temp = weather_df.iloc[i]['temperature']
            weather_mult = 1 + config['weather_sensitivity'] * (temp - 20) / 20
            weather_mult = max(0.7, min(1.5, weather_mult))
            
            # Promotion effect
            promo_mult = config['promo_lift'] if promos[i] else 1.0
            
            # Calculate demand
            demand[i] = (base * weekly_mult * monthly_mult * trend_mult * 
                        weather_mult * promo_mult)
            
            # Add noise
            demand[i] *= np.random.uniform(0.85, 1.15)
            demand[i] = max(1, round(demand[i]))
        
        # Add drift for testing
        if add_drift and random.random() > 0.5:
            drift_type = random.choice(['gradual', 'sudden', 'seasonal_shift'])
            demand = self._add_drift(demand, drift_type=drift_type)
        
        # Create dataframe
        df = pd.DataFrame({
            'date': self.dates,
            'sku': sku,
            'units_sold': demand.astype(int),
            'price': config['price'],
            'promo_flag': promos.astype(int)
        })
        
        return df
    
    def generate_full_dataset(self, skus: Optional[List[str]] = None,
                              add_drift: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with all SKUs."""
        if skus is None:
            skus = list(self.SKU_CONFIGS.keys())
        
        weather_df = self._generate_weather()
        
        all_data = []
        for sku in skus:
            sku_df = self.generate_sku_data(sku, weather_df, add_drift)
            all_data.append(sku_df)
        
        pos_df = pd.concat(all_data, ignore_index=True)
        pos_df = pos_df.sort_values(['date', 'sku']).reset_index(drop=True)
        
        return pos_df, weather_df


def generate_and_save_realistic_data(base_path: str, num_days: int = 365,
                                     add_drift: bool = True):
    """Generate realistic data and save to files."""
    from pathlib import Path
    
    generator = RealisticDataGenerator(
        start_date='2024-01-01',
        num_days=num_days
    )
    
    pos_df, weather_df = generator.generate_full_dataset(add_drift=add_drift)
    
    # Save files
    data_path = Path(base_path) / 'data'
    data_path.mkdir(exist_ok=True)
    
    pos_df.to_csv(data_path / 'pos_data.csv', index=False)
    weather_df.to_csv(data_path / 'external_data.csv', index=False)
    
    # Update business rules for new SKUs
    config_path = Path(base_path) / 'config'
    config_path.mkdir(exist_ok=True)
    
    rules = {'skus': []}
    for sku, config in generator.SKU_CONFIGS.items():
        rules['skus'].append({
            'sku': sku,
            'max_shelf_capacity': int(config['base_demand'] * 3),
            'unit_cost': config['price'],
            'max_budget_per_order': int(config['price'] * config['base_demand'] * 2),
            'perishability_days': config['perishability_days'],
            'safety_stock_days': 2
        })
    
    import yaml
    with open(config_path / 'business_rules.yml', 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)
    
    print(f"Generated {len(pos_df)} POS records for {pos_df['sku'].nunique()} SKUs")
    print(f"Date range: {pos_df['date'].min()} to {pos_df['date'].max()}")
    print(f"Weather records: {len(weather_df)}")
    
    return pos_df, weather_df


if __name__ == "__main__":
    from pathlib import Path
    base_path = Path(__file__).parent.parent
    generate_and_save_realistic_data(str(base_path), num_days=365, add_drift=True)
