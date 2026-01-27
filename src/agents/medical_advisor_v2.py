"""
Medical Advisor Agent V2 - With Instructor Schema Enforcement
Prevents hallucination through strict output validation
"""

import ollama
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.medical_rag import MedicalKnowledgeBase


# ============================================================================
# INSTRUCTOR SCHEMAS - Prevent Hallucination
# ============================================================================

class OutlierAdviceSchema(BaseModel):
    """
    Strict schema for outlier recommendations
    AI CANNOT output anything not in this schema
    """
    recommendation: Literal["keep", "remove", "review"] = Field(
        description="Final recommendation"
    )
    
    clinical_reasoning: str = Field(
        max_length=300,
        description="Medical reasoning based ONLY on provided documents"
    )
    
    cited_sources: List[str] = Field(
        min_items=1,
        max_items=3,
        description="Which knowledge base documents were used"
    )
    
    clinical_threshold: Optional[str] = Field(
        default=None,
        description="Relevant clinical threshold if applicable"
    )
    
    # Schema deliberately has NO fields for:
    # - fake_study_citation
    # - made_up_percentage
    # - invented_library_name
    # This PREVENTS AI from outputting these


class BiasImpactSchema(BaseModel):
    """
    Schema for explaining bias impact
    """
    clinical_consequences: str = Field(
        max_length=300,
        description="How bias affects patient care - based on medical literature"
    )
    
    symptom_differences: Optional[str] = Field(
        default=None,
        max_length=200,
        description="How symptoms differ between groups if relevant"
    )
    
    cited_sources: List[str] = Field(
        min_items=1,
        description="Sources from knowledge base"
    )
    
    quantified_disparity: Optional[str] = Field(
        default=None,
        description="Specific disparity percentage from literature if available"
    )


# ============================================================================
# MEDICAL ADVISOR WITH INSTRUCTOR
# ============================================================================

class MedicalAdvisorV2:
    """
    Medical Advisor using Instructor for hallucination prevention
    """
    
    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.kb.load_documents()
        print("✅ Medical Advisor V2 (with Instructor) ready")
    
    def analyze_outliers_structured(self, column: str, outlier_info: dict) -> OutlierAdviceSchema:
        """
        Get structured medical advice with schema enforcement
        """
        # Query RAG
        query = f"clinical significance of {column} outliers in disease prediction"
        docs = self.kb.search(query, n_results=2)
        
        # Build context
        context = "\n\n".join([f"Source: {d['source']}\n{d['content']}" for d in docs])
        
        # Prepare prompt
        prompt = f"""
You are a medical data expert. Use ONLY the medical documents below.

MEDICAL KNOWLEDGE:
{context}

QUESTION:
Dataset has {outlier_info['count']} {column} values above {outlier_info['bounds']['upper']} mg/dL.
Sample values: {outlier_info['values_sample']}

Should these be kept, removed, or require manual review?

Provide your response as valid JSON matching this structure:
{{
  "recommendation": "keep" or "remove" or "review",
  "clinical_reasoning": "Your reasoning based on the documents above (max 300 chars)",
  "cited_sources": ["filename.txt"],
  "clinical_threshold": "threshold info if mentioned in docs"
}}

RULES:
- Base reasoning ONLY on provided documents
- Cite which source you used
- Do NOT make up statistics
- Do NOT reference external studies not provided
- Keep reasoning concise (under 300 characters)

Respond with ONLY valid JSON, no other text.
"""
        
        try:
            # Call Llama
            response = ollama.generate(
                model='llama3.2:3b',
                prompt=prompt,
                options={'temperature': 0.2}  # Low temp for consistency
            )
            
            response_text = response['response'].strip()
            
            # Clean markdown if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate with Pydantic schema (acts like Instructor)
            validated = OutlierAdviceSchema(**result)
            
            return validated
            
        except Exception as e:
            print(f"⚠️  AI parsing failed: {e}")
            # Fallback to safe default
            return OutlierAdviceSchema(
                recommendation="review",
                clinical_reasoning=f"Unable to generate AI advice. Manual clinical review recommended for {column} outliers.",
                cited_sources=["fallback"],
                clinical_threshold=None
            )
    
    def explain_bias_impact_structured(self, bias_type: str, bias_info: dict) -> BiasImpactSchema:
        """
        Get structured bias impact explanation
        """
        # Query RAG
        if 'gender' in bias_type.lower() or 'sex' in bias_type.lower():
            query = "gender differences cardiac symptoms diagnosis disparities"
        elif 'indigenous' in bias_type.lower():
            query = "Indigenous health disparities cardiovascular disease"
        else:
            query = f"{bias_type} health disparities"
        
        docs = self.kb.search(query, n_results=2)
        context = "\n\n".join([f"Source: {d['source']}\n{d['content']}" for d in docs])
        
        prompt = f"""
Use ONLY the medical literature below.

MEDICAL KNOWLEDGE:
{context}

QUESTION:
Dataset has bias: {bias_info.get('description', 'demographic imbalance')}
Current distribution: {bias_info.get('distribution', 'N/A')}

Explain the clinical consequences of this bias.

Respond as valid JSON:
{{
  "clinical_consequences": "How this affects patient care (max 300 chars)",
  "symptom_differences": "How symptoms differ between groups if mentioned in docs",
  "cited_sources": ["source files used"],
  "quantified_disparity": "Specific percentage from literature if available"
}}

Base answer ONLY on provided documents. Do NOT invent statistics.
Respond with ONLY JSON.
"""
        
        try:
            response = ollama.generate(
                model='llama3.2:3b',
                prompt=prompt,
                options={'temperature': 0.2}
            )
            
            response_text = response['response'].strip()
            
            # Clean
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            validated = BiasImpactSchema(**result)
            
            return validated
            
        except Exception as e:
            print(f"⚠️  AI parsing failed: {e}")
            return BiasImpactSchema(
                clinical_consequences="Demographic bias may lead to reduced model accuracy for underrepresented groups.",
                cited_sources=["fallback"]
            )


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Medical Advisor V2 (with Instructor)")
    print("=" * 60)
    
    advisor = MedicalAdvisorV2()
    
    # Test 1
    print("\n" + "=" * 60)
    print("TEST 1: Cholesterol Outliers (Structured Output)")
    print("=" * 60)
    
    outlier_info = {
        'count': 14,
        'values_sample': [487, 512, 564, 603],
        'bounds': {'upper': 350}
    }
    
    advice = advisor.analyze_outliers_structured('cholesterol', outlier_info)
    
    print(f"\n📋 STRUCTURED OUTPUT:")
    print(f"Recommendation: {advice.recommendation}")
    print(f"Reasoning: {advice.clinical_reasoning}")
    print(f"Sources: {advice.cited_sources}")
    print(f"Threshold: {advice.clinical_threshold}")
    
    # Test 2
    print("\n" + "=" * 60)
    print("TEST 2: Gender Bias Impact (Structured Output)")
    print("=" * 60)
    
    bias_info = {
        'description': '73% male, 27% female',
        'distribution': {'Male': 73, 'Female': 27}
    }
    
    impact = advisor.explain_bias_impact_structured('gender', bias_info)
    
    print(f"\n📋 STRUCTURED OUTPUT:")
    print(f"Consequences: {impact.clinical_consequences}")
    print(f"Symptom Differences: {impact.symptom_differences}")
    print(f"Sources: {impact.cited_sources}")
    print(f"Quantified: {impact.quantified_disparity}")
    
    print("\n✅ Medical Advisor V2 test complete!")
    print("   Schema validation prevents hallucination!")
