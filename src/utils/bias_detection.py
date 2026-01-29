"""
Bias Detector - Fairness Analysis Engine
Detects demographic imbalance and calculates fairness metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class BiasDetector:
    """
    Analyzes dataset for demographic bias and fairness issues
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df
        self.target_col = target_col
        self.bias_issues = {}
        
        # Population benchmarks
        self.benchmarks = {
            'sex': {'Male': 50, 'Female': 50},
            'gender': {'Male': 50, 'Female': 50},
            'race_ethnicity': {
                'White': 60, 'Black': 13, 'Hispanic': 18,
                'Asian': 6, 'Indigenous': 1.3, 'Other': 1.7
            }
        }
    
    def run_full_analysis(self) -> Dict:
        """Run all bias checks"""
        self.analyze_demographic_balance()
        return self.bias_issues
    
    def _normalize_sex_gender_values(self, series: pd.Series) -> pd.Series:
        """
        Normalize sex/gender values to standard format.
        Maps: Male/M/male/1 -> Male, Female/F/female/0 -> Female
        """
        # Convert to string for consistent processing
        normalized = series.astype(str).str.strip()
        
        # Define all possible male and female variants (case-insensitive)
        male_variants = ['male', 'm', '1', '1.0', '1.00', '1.000']
        female_variants = ['female', 'f', '0', '0.0', '0.00', '0.000']
        
        # Create mapping dictionary for all unique values
        mapping = {}
        for val in normalized.unique():
            val_lower = str(val).lower().strip()
            # Handle NaN/null values
            if pd.isna(val) or val_lower in ['nan', 'none', 'null', '']:
                mapping[val] = val  # Keep as-is
            elif val_lower in male_variants:
                mapping[val] = 'Male'
            elif val_lower in female_variants:
                mapping[val] = 'Female'
            else:
                # Keep original if not recognized (might be other categories)
                mapping[val] = val
        
        return normalized.map(mapping)
    
    def analyze_demographic_balance(self):
        """Check demographic representation"""
        demographic_cols = [c for c in self.df.columns if any(k in c.lower() for k in ['sex', 'gender', 'race', 'ethnicity'])]
        
        for col in demographic_cols:
            # Normalize sex/gender columns before calculating distribution
            if 'sex' in col.lower() or 'gender' in col.lower():
                normalized_series = self._normalize_sex_gender_values(self.df[col])
                distribution = normalized_series.value_counts(normalize=True) * 100
            else:
                distribution = self.df[col].value_counts(normalize=True) * 100
            
            benchmark = self.benchmarks.get(col.lower())
            
            issues = []
            # Convert Series to list of values - use .values property (not method)
            dist_values = list(distribution.values)
            
            # For binary distributions (sex/gender), check if gap > 5% (FDA threshold)
            if 'sex' in col.lower() or 'gender' in col.lower():
                if len(dist_values) == 2:
                    gap = abs(dist_values[0] - dist_values[1])
                    if gap > 5:  # FDA threshold for binary balance
                        max_pct = max(dist_values)
                        min_pct = min(dist_values)
                        max_group = distribution.idxmax()
                        min_group = distribution.idxmin()
                        issues.append(f"{min_group}: {min_pct:.1f}% (underrepresented, gap: {gap:.1f}%)")
                        issues.append(f"{max_group}: {max_pct:.1f}% (overrepresented)")
                else:
                    # More than 2 groups - flag any < 20%
                    for group, pct in distribution.items():
                        if pct < 20:
                            issues.append(f"{group}: {pct:.1f}% (underrepresented)")
            else:
                # For multi-group distributions (race/ethnicity), check against benchmarks
                if benchmark:
                    for group, expected_pct in benchmark.items():
                        actual_pct = distribution.get(group, 0)
                        if actual_pct == 0:
                            issues.append(f"{group}: 0% MISSING (expected: {expected_pct}%)")
                        elif actual_pct < expected_pct * 0.5:  # Less than 50% of expected
                            issues.append(f"{group}: {actual_pct:.1f}% (expected: {expected_pct}%, underrepresented)")
                else:
                    # No benchmark - flag any group < 20%
                    for group, pct in distribution.items():
                        if pct < 20:
                            issues.append(f"{group}: {pct:.1f}% (underrepresented)")
            
            self.bias_issues[col] = {
                'distribution': distribution.to_dict(),
                'bias_detected': len(issues) > 0,
                'issues': issues
            }
    
    def get_summary(self) -> Dict:
        """Get summary"""
        total_bias = sum(1 for i in self.bias_issues.values() if i.get('bias_detected'))
        return {
            'attributes_analyzed': len(self.bias_issues),
            'attributes_with_bias': total_bias,
            'status': 'BIASED' if total_bias > 0 else 'FAIR'
        }


if __name__ == "__main__":
    print("Testing Bias Detector...\n")
    
    # Test Dataset 2
    df2 = pd.read_csv('data/synthetic/dataset2_diabetes_gender_bias.csv')
    detector2 = BiasDetector(df2, 'diabetes')
    issues2 = detector2.run_full_analysis()
    
    print("\nDataset 2 Results:")
    print(f"Bias detected: {detector2.get_summary()}")
    
    # Test Dataset 3
    df3 = pd.read_csv('data/synthetic/dataset3_heart_disease_indigenous.csv')
    detector3 = BiasDetector(df3, 'heart_disease')
    issues3 = detector3.run_full_analysis()
    
    print("\nDataset 3 Results:")
    print(f"Bias detected: {detector3.get_summary()}")
    
    print("\n✅ Complete!")
