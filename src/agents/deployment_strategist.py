"""
Deployment Strategist Agent - Creates week-by-week implementation plans
Synthesizes all findings into actionable roadmap
"""

import ollama
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json


# ============================================================================
# SCHEMAS
# ============================================================================

class WeeklyTask(BaseModel):
    """Tasks for a specific week"""
    week_number: int = Field(ge=1, le=52)
    days_range: str = Field(description="e.g., 'Days 1-7'")
    tasks: List[str] = Field(min_length=1, max_length=8)
    deliverables: List[str] = Field(min_length=1, max_length=5)
    estimated_hours: Optional[int] = Field(default=None, ge=1, le=168)


class AlternativePath(BaseModel):
    """Alternative strategy if constraints change"""
    scenario: str = Field(max_length=150)
    when_to_pivot: str = Field(max_length=150)
    modified_timeline: str = Field(max_length=300)


class DeploymentPlanSchema(BaseModel):
    """
    Complete deployment roadmap with week-by-week tasks
    """
    recommended_strategy: str = Field(
        max_length=200,
        description="High-level strategy name"
    )
    
    why_this_strategy: str = Field(
        max_length=300,
        description="Why this fits user's constraints"
    )
    
    weekly_plan: List[WeeklyTask] = Field(
        min_length=1,
        max_length=12,
        description="Week-by-week breakdown"
    )
    
    critical_path_items: List[str] = Field(
        min_length=1,
        max_length=5,
        description="Must-complete items for success"
    )
    
    risk_factors: List[str] = Field(
        min_length=1,
        max_length=5,
        description="What could go wrong"
    )
    
    alternative_paths: List[AlternativePath] = Field(
        max_length=3,
        description="Backup plans if situation changes"
    )
    
    success_metrics: List[str] = Field(
        min_length=2,
        max_length=5,
        description="How to measure success"
    )


# ============================================================================
# DEPLOYMENT STRATEGIST
# ============================================================================

class DeploymentStrategist:
    """
    Creates actionable implementation roadmaps
    """
    
    def __init__(self):
        print("✅ Deployment Strategist Agent initialized")
    
    def create_plan(
        self,
        quality_issues: Dict,
        bias_issues: Dict,
        fairness_recommendations: Dict,
        user_context: Dict
    ) -> DeploymentPlanSchema:
        """
        Synthesize everything into week-by-week action plan
        """
        
        prompt = f"""
You are a project strategist creating an implementation roadmap.

DATA QUALITY ISSUES:
{json.dumps(quality_issues, indent=2)}

BIAS ISSUES:
{json.dumps(bias_issues, indent=2)}

USER CONSTRAINTS:
- Timeline: {user_context.get('timeline_days', 90)} days
- Model: {user_context.get('model_type', 'Unknown')}
- Use case: {user_context.get('use_case', 'Unknown')}
- Can collect data: {user_context.get('can_collect_data', 'Unknown')}

FAIRNESS RECOMMENDATIONS:
- Samples needed: {fairness_recommendations.get('exact_samples_needed', {})}
- Recruitment timeline: {fairness_recommendations.get('timeline_months', 'N/A')} months
- Immediate fix: {fairness_recommendations.get('immediate_method', 'SMOTE')}

Create a SPECIFIC week-by-week plan as JSON:

{{
  "recommended_strategy": "Brief strategy name",
  "why_this_strategy": "Why this fits user's {user_context.get('timeline_days')} day timeline and {user_context.get('use_case')} use case",
  
  "weekly_plan": [
    {{
      "week_number": 1,
      "days_range": "Days 1-7",
      "tasks": ["Specific task 1", "Specific task 2"],
      "deliverables": ["Deliverable 1"],
      "estimated_hours": 20  // MUST be an integer >= 1 (not a float like 2.5)
    }},
    // Continue for user's timeline
  ],
  
  "critical_path_items": ["Must-do item 1", "Must-do item 2"],
  "risk_factors": ["Risk 1", "Risk 2"],
  
  "alternative_paths": [
    {{
      "scenario": "If can recruit only 200 patients instead of 552",
      "when_to_pivot": "If recruitment slower than expected",
      "modified_timeline": "Use hybrid approach: 200 real + 352 SMOTE"
    }}
  ],
  
  "success_metrics": ["Metric 1", "Metric 2"]
}}

Make tasks SPECIFIC and ACTIONABLE. Include exact numbers.
IMPORTANT: estimated_hours must be an INTEGER >= 1 (not a float). Use whole numbers like 8, 16, 20, not 2.5 or 1.75.
Respond with ONLY valid JSON.
"""
        
        try:
            response = ollama.generate(
                model='llama3.2:3b',
                prompt=prompt,
                options={'temperature': 0.2, 'num_predict': 1500}
            )
            
            response_text = response['response'].strip()
            
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            
            # Sanitize estimated_hours: convert floats to ints, ensure >= 1
            if 'weekly_plan' in result:
                for week in result['weekly_plan']:
                    if 'estimated_hours' in week:
                        hours = week['estimated_hours']
                        if hours is not None:
                            # Convert float to int (round)
                            if isinstance(hours, float):
                                hours = int(round(hours))
                            # Ensure >= 1, otherwise set to None
                            if hours < 1:
                                hours = None
                            week['estimated_hours'] = hours
            
            validated = DeploymentPlanSchema(**result)
            
            print("✅ Deployment plan generated")
            return validated
            
        except Exception as e:
            print(f"⚠️  Plan generation failed: {e}")
            return self._create_fallback_plan(user_context)
    
    def _create_fallback_plan(self, user_context: Dict) -> DeploymentPlanSchema:
        """Safe fallback plan"""
        
        timeline_weeks = user_context.get('timeline_days', 90) // 7
        
        return DeploymentPlanSchema(
            recommended_strategy="Data quality fixes then model training",
            why_this_strategy="Address critical issues before model development",
            weekly_plan=[
                WeeklyTask(
                    week_number=1,
                    days_range="Days 1-7",
                    tasks=["Fix missing values", "Remove duplicates", "Apply bias mitigation"],
                    deliverables=["Cleaned dataset"],
                    estimated_hours=20
                )
            ],
            critical_path_items=["Data cleaning", "Bias mitigation"],
            risk_factors=["Timeline may be too short for data collection"],
            alternative_paths=[],
            success_metrics=["Dataset quality score >0.9", "Fairness metrics <5% disparity"]
        )


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Deployment Strategist Agent")
    print("=" * 60)
    
    strategist = DeploymentStrategist()
    
    # Mock inputs
    quality_issues = {
        'missing_cholesterol': {'count': 172, 'percentage': 17.2},
        'duplicates': 23
    }
    
    bias_issues = {
        'gender_bias': {'Male': 73.0, 'Female': 27.0}
    }
    
    fairness_rec = {
        'exact_samples_needed': {'Female': 552},
        'timeline_months': 6,
        'immediate_method': 'SMOTE'
    }
    
    user_context = {
        'timeline_days': 60,
        'model_type': 'Random Forest',
        'use_case': 'clinical_deployment',
        'can_collect_data': 'no',
        'location': 'Boston'
    }
    
    # Generate plan
    print("\n🤖 Generating deployment roadmap...")
    plan = strategist.create_plan(quality_issues, bias_issues, fairness_rec, user_context)
    
    # Display
    print("\n" + "=" * 60)
    print("📅 DEPLOYMENT ROADMAP")
    print("=" * 60)
    
    print(f"\n🎯 Strategy: {plan.recommended_strategy}")
    print(f"Why: {plan.why_this_strategy}")
    
    print(f"\n📅 Week-by-Week Plan:")
    for week in plan.weekly_plan:
        print(f"\n  WEEK {week.week_number} ({week.days_range}):")
        print(f"  Tasks:")
        for task in week.tasks:
            print(f"    • {task}")
        print(f"  Deliverables: {', '.join(week.deliverables)}")
    
    print(f"\n🎯 Critical Path: {', '.join(plan.critical_path_items)}")
    print(f"\n⚠️  Risks: {', '.join(plan.risk_factors)}")
    print(f"\n📊 Success Metrics: {', '.join(plan.success_metrics)}")
    
    print("\n✅ Deployment plan complete!")
