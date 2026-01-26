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
        print("⚖️ Running bias detection...")
        self.analyze_demographic_balance()
        return self.bias_issues
    
    def analyze_demographic_balance(self):
        """Check demographic representation"""
        demographic_cols = [c for c in self.df.columns if any(k in c.lower() for k in ['sex', 'gender', 'race', 'ethnicity'])]
        
        for col in demographic_cols:
            distribution = self.df[col].value_counts(normalize=True) * 100
            benchmark = self.benchmarks.get(col.lower())
            
            issues = []
            for group, pct in distribution.items():
                if pct < 20:
                    issues.append(f"{group}: {pct:.1f}%")
            
            # Check for missing groups
            if benchmark:
                for group in benchmark.keys():
                    if group not in distribution.index:
                        issues.append(f"{group}: 0% MISSING")
            
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
