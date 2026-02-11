"""
Drift Diagnostics Module
Generates human-readable reports explaining drift causes and recommendations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosticsEngine:
    """Generates diagnostic reports for drift events."""
    
    def __init__(self):
        self.reports: List[Dict] = []
    
    def analyze_residuals(self, residual_history: List[float], 
                         lookback: int = 14) -> Dict:
        """Analyze residual patterns for diagnostics."""
        if len(residual_history) < lookback:
            return {'status': 'insufficient_data'}
        
        recent = np.array(residual_history[-lookback//2:])
        historical = np.array(residual_history[-lookback:-lookback//2])
        
        recent_mae = np.mean(np.abs(recent))
        historical_mae = np.mean(np.abs(historical))
        
        recent_std = np.std(recent)
        historical_std = np.std(historical)
        
        mae_change = ((recent_mae - historical_mae) / historical_mae * 100) if historical_mae > 0 else 0
        std_change = ((recent_std - historical_std) / historical_std * 100) if historical_std > 0 else 0
        
        return {
            'recent_mae': round(recent_mae, 2),
            'historical_mae': round(historical_mae, 2),
            'mae_change_pct': round(mae_change, 1),
            'recent_std': round(recent_std, 2),
            'historical_std': round(historical_std, 2),
            'std_change_pct': round(std_change, 1)
        }
    
    def analyze_features(self, before_df: pd.DataFrame, after_df: pd.DataFrame,
                        features: List[str]) -> List[Dict]:
        """Analyze feature distribution changes."""
        changes = []
        
        for feature in features:
            if feature not in before_df.columns or feature not in after_df.columns:
                continue
            
            before_vals = before_df[feature].dropna()
            after_vals = after_df[feature].dropna()
            
            if len(before_vals) == 0 or len(after_vals) == 0:
                continue
            
            before_mean = before_vals.mean()
            after_mean = after_vals.mean()
            
            change_pct = ((after_mean - before_mean) / before_mean * 100) if before_mean != 0 else 0
            
            if abs(change_pct) > 20:  # Significant change threshold
                changes.append({
                    'feature': feature,
                    'before_mean': round(before_mean, 2),
                    'after_mean': round(after_mean, 2),
                    'change_pct': round(change_pct, 1),
                    'significance': 'high' if abs(change_pct) > 50 else 'medium'
                })
        
        return sorted(changes, key=lambda x: abs(x['change_pct']), reverse=True)
    
    def get_recommendations(self, drift_severity: float, 
                           feature_changes: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on diagnostics."""
        recommendations = []
        
        if drift_severity > 0.7:
            recommendations.append("🔴 URGENT: Immediate model retraining recommended")
        elif drift_severity > 0.4:
            recommendations.append("🟡 Model retraining recommended within 24-48 hours")
        else:
            recommendations.append("🟢 Continue monitoring; scheduled retraining sufficient")
        
        high_impact_features = [f for f in feature_changes if f.get('significance') == 'high']
        if high_impact_features:
            features_str = ', '.join([f['feature'] for f in high_impact_features[:3]])
            recommendations.append(f"Review changes in: {features_str}")
        
        promo_change = next((f for f in feature_changes if 'promo' in f['feature'].lower()), None)
        if promo_change and promo_change['change_pct'] > 30:
            recommendations.append("Check promotion calendar - significant promo pattern change detected")
        
        return recommendations


def generate_drift_report(sku_id: str, drift_info: Dict,
                         feature_stats_before: Optional[Dict] = None,
                         feature_stats_after: Optional[Dict] = None) -> str:
    """
    Generate a markdown-formatted drift diagnostic report.
    
    Args:
        sku_id: SKU identifier
        drift_info: Drift detection results
        feature_stats_before: Feature statistics before drift window
        feature_stats_after: Feature statistics after drift window
        
    Returns:
        Markdown-formatted diagnostic report
    """
    lines = []
    
    # Header
    lines.append(f"# Drift Diagnostic Report: {sku_id}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    severity = drift_info.get('severity', 0)
    severity_label = "🔴 High" if severity > 0.7 else "🟡 Medium" if severity > 0.4 else "🟢 Low"
    
    lines.append("## Summary")
    lines.append(f"- **Drift Detected:** {'Yes' if drift_info.get('drift_detected') else 'No'}")
    lines.append(f"- **Severity:** {severity_label} ({severity:.0%})")
    lines.append(f"- **Detectors Triggered:** {', '.join(drift_info.get('detectors_triggered', ['None']))}")
    lines.append("")
    
    # Residual Statistics
    residual_stats = drift_info.get('residual_stats', {})
    if residual_stats:
        lines.append("## Error Statistics")
        lines.append(f"- Mean Absolute Error: {residual_stats.get('mae', 0):.2f}")
        lines.append(f"- Error Std Dev: {residual_stats.get('std', 0):.2f}")
        lines.append(f"- Sample Count: {residual_stats.get('count', 0)}")
        lines.append("")
    
    # Input Drift
    input_drift = drift_info.get('input_drift', [])
    if input_drift:
        lines.append("## Feature Distribution Changes")
        for fd in input_drift:
            status = "⚠️ Shifted" if fd.get('drift_detected') else "✓ Stable"
            lines.append(f"- **{fd.get('feature', 'Unknown')}:** {status} (p={fd.get('p_value', 'N/A')})")
        lines.append("")
    
    # Feature Changes
    if feature_stats_before and feature_stats_after:
        lines.append("## Suspected Causes")
        for key in feature_stats_after:
            if key in feature_stats_before:
                before_val = feature_stats_before[key]
                after_val = feature_stats_after[key]
                if before_val != 0:
                    change = ((after_val - before_val) / before_val) * 100
                    if abs(change) > 20:
                        lines.append(f"- **{key}:** {before_val:.2f} → {after_val:.2f} ({change:+.1f}%)")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommended Actions")
    if severity > 0.7:
        lines.append("1. **Retrain model immediately** with recent data")
        lines.append("2. Investigate external factors (promotions, seasonality, supply issues)")
    elif severity > 0.4:
        lines.append("1. Schedule model retraining within 24-48 hours")
        lines.append("2. Review business rules for this SKU")
    else:
        lines.append("1. Continue monitoring")
        lines.append("2. Include in next scheduled retraining cycle")
    
    return '\n'.join(lines)


def summarize_drift_events(events: List[Dict]) -> pd.DataFrame:
    """Convert drift events to summary DataFrame."""
    if not events:
        return pd.DataFrame(columns=['timestamp', 'sku', 'severity', 'detectors'])
    
    rows = []
    for event in events:
        rows.append({
            'timestamp': event.get('timestamp', ''),
            'sku': event.get('sku', ''),
            'severity': event.get('severity', 0),
            'detectors': ', '.join(event.get('detectors', []))
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test diagnostics
    drift_info = {
        'drift_detected': True,
        'severity': 0.65,
        'detectors_triggered': ['ADWIN', 'DDM'],
        'residual_stats': {'mae': 12.5, 'std': 8.3, 'count': 50},
        'input_drift': [
            {'feature': 'promo_flag', 'drift_detected': True, 'p_value': 0.02},
            {'feature': 'temperature', 'drift_detected': False, 'p_value': 0.45}
        ]
    }
    
    report = generate_drift_report("MILK_1L", drift_info)
    print(report)
