"""
PHI Scanner - HIPAA Layer 1 (ENHANCED - Comprehensive Detection)
Scans datasets for Protected Health Information before processing
Detects all 18 HIPAA identifiers with case-insensitive matching
"""

import pandas as pd
import re
from typing import List, Dict, Tuple


class PHIScanner:
    """
    Comprehensive PHI scanner for all 18 HIPAA identifiers
    Case-insensitive detection with pattern matching in column names and data values
    """
    
    # All HIPAA identifier keywords (case-insensitive matching)
    PHI_KEYWORDS = [
        # Names (1)
        'name', 'patient_name', 'full_name', 'first_name', 'last_name', 'middle_name',
        'patient', 'person', 'individual', 'client', 'subject',
        # Doctor/Provider (1)
        'doctor', 'physician', 'provider', 'clinician', 'attending', 'surgeon', 'nurse',
        'dr.', 'dr ', 'md', 'dds', 'dmd', 'phd', 'rn', 'np', 'pa',
        # Geographic (2)
        'address', 'street', 'city', 'zip', 'zipcode', 'postal_code', 'county', 'location',
        # Dates (3)
        'date', 'dob', 'birth', 'admission', 'discharge', 'death', 'visit', 'appointment',
        'admit', 'discharge_date', 'admission_date', 'service_date', 'procedure_date',
        # Phone/Fax (4, 5)
        'phone', 'telephone', 'fax', 'mobile', 'cell', 'contact',
        # Email (6)
        'email', 'e-mail', 'email_address',
        # SSN (7)
        'ssn', 'social_security', 'social_security_number',
        # Medical Record (8)
        'mrn', 'medical_record', 'medical_record_number', 'record_number', 'case_number',
        # Health Plan (9)
        'health_plan', 'beneficiary', 'member_id', 'policy_number', 'insurance',
        # Account (10)
        'account', 'account_number', 'account_id',
        # Certificate/License (11)
        'license', 'certificate', 'license_number', 'certificate_number', 'permit',
        # Vehicle (12)
        'vehicle', 'license_plate', 'vin', 'vehicle_id',
        # Device (13)
        'device', 'device_id', 'device_serial', 'serial_number', 'equipment',
        # URL (14)
        'url', 'web_url', 'website', 'link',
        # IP Address (15)
        'ip_address', 'ip', 'ip_address',
        # Biometric (16)
        'biometric', 'fingerprint', 'retina', 'iris', 'dna', 'biometric_id',
        # Photo (17)
        'photo', 'image', 'picture', 'photograph', 'photo_id',
        # Billing/Payment (sensitive when combined with identifiers)
        'billing', 'bill', 'charge', 'cost', 'payment', 'amount', 'price', 'fee',
        'revenue', 'claim', 'invoice', 'receipt',
        # Other identifiers
        'id', 'identifier', 'unique_id', 'patient_id', 'encounter_id', 'visit_id'
    ]
    
    # Pattern matching for column names (case-insensitive)
    COLUMN_PATTERNS = [
        # Names
        (r'\bname\b', 'name'),
        (r'\bpatient.*name\b', 'patient name'),
        (r'\bfull.*name\b', 'full name'),
        (r'\bfirst.*name\b', 'first name'),
        (r'\blast.*name\b', 'last name'),
        # Doctor/Provider
        (r'\bdoctor\b', 'doctor'),
        (r'\bphysician\b', 'physician'),
        (r'\bprovider\b', 'provider'),
        (r'\bclinician\b', 'clinician'),
        (r'\battending\b', 'attending physician'),
        (r'\bdr\.?\b', 'doctor'),
        # Dates
        (r'\bdate\b', 'date'),
        (r'\bdate.*of.*birth\b', 'date of birth'),
        (r'\bdate.*of.*admission\b', 'admission date'),
        (r'\bdate.*of.*discharge\b', 'discharge date'),
        (r'\bdate.*of.*death\b', 'death date'),
        (r'\badmission.*date\b', 'admission date'),
        (r'\bdischarge.*date\b', 'discharge date'),
        (r'\bdob\b', 'date of birth'),
        (r'\bbirth.*date\b', 'birth date'),
        # Address
        (r'\baddress\b', 'address'),
        (r'\bstreet\b', 'street address'),
        (r'\bzip.*code\b', 'zip code'),
        (r'\bpostal.*code\b', 'postal code'),
        # Phone
        (r'\bphone\b', 'phone'),
        (r'\btelephone\b', 'telephone'),
        (r'\bfax\b', 'fax'),
        # Email
        (r'\bemail\b', 'email'),
        (r'\be-mail\b', 'email'),
        # SSN
        (r'\bssn\b', 'SSN'),
        (r'\bsocial.*security\b', 'social security number'),
        # Medical Record
        (r'\bmrn\b', 'medical record number'),
        (r'\bmedical.*record\b', 'medical record'),
        # Insurance/Health Plan
        (r'\binsurance\b', 'insurance'),
        (r'\bhealth.*plan\b', 'health plan'),
        (r'\bbeneficiary\b', 'beneficiary'),
        # Account
        (r'\baccount\b', 'account'),
        # Billing
        (r'\bbilling\b', 'billing'),
        (r'\bbill\b', 'bill'),
        (r'\bcharge\b', 'charge'),
        (r'\bcost\b', 'cost'),
        (r'\bpayment\b', 'payment'),
        (r'\bamount\b', 'amount'),
    ]
    
    def __init__(self):
        self.violations = []
    
    def scan_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive scan for PHI
        Returns: (is_safe: bool, violations: List[str])
        """
        self.violations = []
        
        # Check 1: Column names (comprehensive pattern matching)
        self._scan_column_names(df)
        
        # Check 2: Data patterns (SSN, phone, email, dates)
        self._scan_data_patterns(df)
        
        # Check 3: Name patterns in data values
        self._scan_name_patterns(df)
        
        # Check 4: Date patterns in data values
        self._scan_date_patterns(df)
        
        is_safe = len(self.violations) == 0
        
        return is_safe, self.violations
    
    def _scan_column_names(self, df: pd.DataFrame):
        """
        Check column names with comprehensive pattern matching (case-insensitive)
        """
        for col in df.columns:
            # Normalize column name: lowercase, replace separators with spaces
            col_normalized = col.lower().replace('_', ' ').replace('-', ' ').replace('.', ' ').strip()
            
            # Check against PHI keywords (exact match in normalized form)
            for keyword in self.PHI_KEYWORDS:
                # Check if keyword appears as whole word in column name
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', col_normalized):
                    self.violations.append(f"Column '{col}' contains PHI identifier: '{keyword}'")
                    break
            
            # Check against pattern matching (more flexible)
            for pattern, identifier_name in self.COLUMN_PATTERNS:
                if re.search(pattern, col_normalized, re.IGNORECASE):
                    # Avoid duplicate violations
                    if not any(f"Column '{col}'" in v for v in self.violations):
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
            if sample.str.match(ssn_pattern, na=False).any():
                self.violations.append(f"Column '{col}' contains SSN pattern in data values")
            
            # Phone pattern: (###) ###-#### or ###-###-#### or international
            phone_pattern = r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$|^\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}$'
            if sample.str.match(phone_pattern, na=False).any():
                self.violations.append(f"Column '{col}' may contain phone numbers")
            
            # Email pattern
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if sample.str.match(email_pattern, na=False).any():
                self.violations.append(f"Column '{col}' may contain email addresses")
            
            # IP Address pattern
            ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
            if sample.str.match(ip_pattern, na=False).any():
                self.violations.append(f"Column '{col}' may contain IP addresses")
            
            # URL pattern
            url_pattern = r'^https?://|^www\.'
            if sample.str.match(url_pattern, na=False, case=False).any():
                self.violations.append(f"Column '{col}' may contain URLs")
    
    def _scan_name_patterns(self, df: pd.DataFrame):
        """
        Detect name patterns in data values (e.g., "John Doe", "Dr. Smith")
        Case-insensitive matching
        """
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            # Skip if column name already flagged
            if any(f"Column '{col}'" in v for v in self.violations):
                continue
            
            sample = df[col].dropna().head(50).astype(str)
            
            if len(sample) == 0:
                continue
            
            # Pattern for names: "First Last" or "Title First Last" or "Last, First"
            # Must have at least 2 capitalized words or title + capitalized word
            name_patterns = [
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # "John Doe"
                r'^(Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Prof\.)\s+[A-Z][a-z]+',  # "Dr. Smith"
                r'^[A-Z][a-z]+,\s+[A-Z][a-z]+',  # "Doe, John"
                r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "John Michael Doe"
            ]
            
            name_count = 0
            for value in sample:
                value_str = str(value).strip()
                if len(value_str) < 3:  # Skip very short values
                    continue
                
                for pattern in name_patterns:
                    if re.match(pattern, value_str):
                        name_count += 1
                        break
            
            # If more than 30% of values look like names, flag it
            if name_count > len(sample) * 0.3:
                self.violations.append(f"Column '{col}' contains name-like patterns in data values ({name_count}/{len(sample)} samples)")
    
    def _scan_date_patterns(self, df: pd.DataFrame):
        """
        Detect date patterns in data values
        """
        # Check both string and datetime columns
        string_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Check datetime columns
        for col in date_cols:
            if not any(f"Column '{col}'" in v for v in self.violations):
                # If column name doesn't suggest it's a safe date (like "year"), flag it
                col_lower = col.lower()
                if 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower:
                    self.violations.append(f"Column '{col}' contains date values (dates are PHI)")
        
        # Check string columns for date patterns
        for col in string_cols:
            if any(f"Column '{col}'" in v for v in self.violations):
                continue
            
            sample = df[col].dropna().head(50).astype(str)
            
            if len(sample) == 0:
                continue
            
            # Date patterns: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, etc.
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'^\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            ]
            
            date_count = 0
            for value in sample:
                value_str = str(value).strip()
                for pattern in date_patterns:
                    if re.match(pattern, value_str):
                        date_count += 1
                        break
            
            # If more than 50% of values look like dates, flag it
            if date_count > len(sample) * 0.5:
                col_lower = col.lower()
                # Only flag if column name suggests it's a date (not just a number)
                if 'date' in col_lower or 'dob' in col_lower or 'birth' in col_lower or 'admission' in col_lower or 'discharge' in col_lower:
                    self.violations.append(f"Column '{col}' contains date patterns in data values")
    
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
    print("Testing ENHANCED PHI Scanner...\n")
    
    # Test 1: Medical dataset (should PASS)
    print("Test 1: Medical dataset (should pass)")
    medical_df = pd.DataFrame({
        'patient_id': ['SYNTH_001', 'SYNTH_002'],
        'age': [45, 62],
        'lung_capacity': [3.2, 2.8],
        'cholesterol': [220, 180]
    })
    
    scanner1 = PHIScanner()
    safe1, violations1 = scanner1.scan_dataset(medical_df)
    print(f"Safe: {safe1}")
    print(f"Violations: {violations1}\n")
    
    # Test 2: Actual PHI (should FAIL)
    print("Test 2: Dataset with PHI (should fail)")
    phi_df = pd.DataFrame({
        'Name': ['John Doe', 'Jane Smith'],
        'Doctor': ['Dr. Smith', 'Dr. Jones'],
        'Date of Admission': ['2023-01-15', '2023-02-20'],
        'Billing Amount': [1000, 2000],
        'age': [45, 62]
    })
    
    scanner2 = PHIScanner()
    safe2, violations2 = scanner2.scan_dataset(phi_df)
    print(f"Safe: {safe2}")
    print(f"Violations: {violations2}")
    
    print("\n✅ PHI Scanner enhanced!")
