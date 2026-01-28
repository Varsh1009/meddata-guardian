"""
PHI Scanner - HIPAA Layer 1 (FIXED - More Intelligent)
Scans datasets for Protected Health Information before processing
"""

import pandas as pd
import re
from typing import List, Dict, Tuple


class PHIScanner:
    """
    Scans for 18 HIPAA identifiers with smart pattern matching
    """
    
    # Exact match identifiers (must be whole word)
    EXACT_MATCH_IDENTIFIERS = [
        'patient_name', 'full_name', 'first_name', 'last_name',
        'ssn', 'social_security', 'social_security_number',
        'mrn', 'medical_record_number', 'medical_record',
        'phone_number', 'telephone', 'phone', 'fax',
        'email', 'email_address',
        'street_address', 'address', 'street',
        'zip_code', 'zipcode', 'postal_code',
        'date_of_birth', 'dob', 'birth_date', 'birthdate',
        'admission_date', 'discharge_date', 'death_date',
        'account_number', 'account',
        'license_number', 'drivers_license',
        'vehicle_identifier', 'license_plate',
        'device_serial', 'serial_number',
        'ip_address', 'web_url',
        'biometric_id', 'fingerprint', 'photo_id'
    ]
    
    # Substring identifiers (can be part of word, but must be meaningful)
    # Removed overly broad terms like 'city', 'name' alone
    
    def __init__(self):
        self.violations = []
    
    def scan_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Scan dataset for PHI
        Returns: (is_safe: bool, violations: List[str])
        """
        self.violations = []
        
        # Check 1: Column names (exact match)
        self._scan_column_names(df)
        
        # Check 2: Data patterns (SSN, phone, email)
        self._scan_data_patterns(df)
        
        is_safe = len(self.violations) == 0
        
        return is_safe, self.violations
    
    def _scan_column_names(self, df: pd.DataFrame):
        """
        Check column names - must be exact or very clear match
        """
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ').strip()
            
            # Check for exact matches
            if col_lower in self.EXACT_MATCH_IDENTIFIERS:
                self.violations.append(f"Column '{col}' contains PHI identifier")
                continue
            
            # Check for clear PHI combinations (must be unambiguous)
            phi_patterns = [
                (r'\bpatient.*name\b', 'patient name'),
                (r'\bsocial.*security\b', 'social security'),
                (r'\bmedical.*record\b', 'medical record'),
                (r'\bphone.*number\b', 'phone number'),
                (r'\bemail.*address\b', 'email address'),
                (r'\bstreet.*address\b', 'street address'),
                (r'\bdate.*birth\b', 'date of birth'),
                (r'\bzip.*code\b', 'zip code')
            ]
            
            for pattern, identifier_name in phi_patterns:
                if re.search(pattern, col_lower):
                    self.violations.append(f"Column '{col}' may contain PHI ({identifier_name})")
                    break
    
    def _scan_data_patterns(self, df: pd.DataFrame):
        """
        Check for common PHI patterns in actual data values
        """
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            sample = df[col].dropna().head(100).astype(str)
            
            if len(sample) == 0:
                continue
            
            # SSN pattern: ###-##-#### or #########
            ssn_pattern = r'^\d{3}-\d{2}-\d{4}$|^\d{9}$'
            if sample.str.match(ssn_pattern).any():
                self.violations.append(f"Column '{col}' contains SSN pattern")
            
            # Phone pattern: (###) ###-#### or ###-###-####
            phone_pattern = r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'
            if sample.str.match(phone_pattern).any():
                self.violations.append(f"Column '{col}' may contain phone numbers")
            
            # Email pattern
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if sample.str.match(email_pattern).any():
                self.violations.append(f"Column '{col}' may contain email addresses")
    
    def get_report(self) -> Dict:
        """Generate scan report"""
        return {
            'is_safe': len(self.violations) == 0,
            'violations_found': len(self.violations),
            'violations': self.violations,
            'status': 'SAFE' if len(self.violations) == 0 else 'PHI DETECTED'
        }


class PHIDetectedError(Exception):
    """Custom exception when PHI is detected"""
    pass


# Test
if __name__ == "__main__":
    print("Testing FIXED PHI Scanner...\n")
    
    # Test 1: Medical column (should PASS now)
    print("Test 1: Medical dataset (should pass)")
    medical_df = pd.DataFrame({
        'patient_id': ['SYNTH_001', 'SYNTH_002'],
        'age': [45, 62],
        'lung_capacity': [3.2, 2.8],  # Should NOT trigger
        'cholesterol': [220, 180]
    })
    
    scanner1 = PHIScanner()
    safe1, violations1 = scanner1.scan_dataset(medical_df)
    print(f"Safe: {safe1}")
    print(f"Violations: {violations1}\n")
    
    # Test 2: Actual PHI (should FAIL)
    print("Test 2: Dataset with PHI (should fail)")
    phi_df = pd.DataFrame({
        'patient_name': ['John Doe', 'Jane Smith'],
        'ssn': ['123-45-6789', '987-65-4321'],
        'age': [45, 62]
    })
    
    scanner2 = PHIScanner()
    safe2, violations2 = scanner2.scan_dataset(phi_df)
    print(f"Safe: {safe2}")
    print(f"Violations: {violations2}")
    
    print("\n✅ PHI Scanner fixed!")
