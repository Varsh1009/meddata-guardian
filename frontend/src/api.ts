const API_BASE = import.meta.env.VITE_API_URL || '';

export type UserContext = {
  project_description: string;
  model_type: string;
  use_case: string;
  timeline_days: number;
  can_collect_data: string;
  location: string;
};

export async function listDemos(): Promise<{ demos: string[] }> {
  const r = await fetch(`${API_BASE}/api/demos`);
  if (!r.ok) throw new Error('Failed to list demos');
  return r.json();
}

export async function phiScanDemo(demoName: string) {
  const form = new FormData();
  form.append('demo_name', demoName);
  const r = await fetch(`${API_BASE}/api/phi-scan-demo`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'PHI scan failed');
  }
  return r.json();
}

export async function phiScanFile(file: File) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_BASE}/api/phi-scan`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'PHI scan failed');
  }
  return r.json();
}

export async function generateSynthetic(options: {
  demo_name?: string;
  file?: File;
  n_samples: number;
}) {
  const form = new FormData();
  form.append('n_samples', String(options.n_samples));
  if (options.demo_name) form.append('demo_name', options.demo_name);
  if (options.file) form.append('file', options.file);
  const r = await fetch(`${API_BASE}/api/synthetic/generate`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'Synthetic generation failed');
  }
  return r.json();
}

export async function analysisQuality(data: Record<string, unknown>[]) {
  const r = await fetch(`${API_BASE}/api/analysis/quality`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data }),
  });
  if (!r.ok) throw new Error('Quality analysis failed');
  return r.json();
}

export async function analysisBias(data: Record<string, unknown>[], target_col?: string) {
  const r = await fetch(`${API_BASE}/api/analysis/bias`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data, target_col }),
  });
  if (!r.ok) throw new Error('Bias analysis failed');
  return r.json();
}

export type FairnessRecommendation = {
  bias_type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  current_distribution: Record<string, number>;
  target_distribution: Record<string, number>;
  exact_samples_needed: Record<string, number>;
  predicted_harm: {
    performance_gap_percentage: number;
    patients_affected_per_100: number;
    clinical_consequence: string;
    annual_impact_estimate?: string | null;
  };
  recruitment_plan: {
    target_facilities: string[];
    timeline_months: number;
    estimated_cost_usd?: number | null;
    specific_contacts?: string | null;
  };
  immediate_technical_fix: {
    method: string;
    python_code: string;
    expected_improvement: string;
    limitations: string;
  };
  fits_user_timeline: boolean;
  recommendation_priority: string;
};

export async function fairnessSpecialistAnalyze(body: {
  attribute: string;
  distribution: Record<string, number>;
  issues: string[];
  user_context: Record<string, unknown>;
  total_samples: number;
}): Promise<FairnessRecommendation> {
  const r = await fetch(`${API_BASE}/api/fairness/specialist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(typeof err.detail === 'string' ? err.detail : 'Fairness specialist failed');
  }
  return r.json();
}

export async function getRecommendations(quality_issues: Record<string, unknown>, bias_issues: Record<string, unknown>) {
  const r = await fetch(`${API_BASE}/api/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ quality_issues, bias_issues }),
  });
  if (!r.ok) throw new Error('Failed to get recommendations');
  return r.json();
}

export async function implementChanges(
  data: Record<string, unknown>[],
  selected_keys: string[],
  quality_issues: Record<string, unknown>,
  recommendations: Record<string, unknown>[]
) {
  const r = await fetch(`${API_BASE}/api/implement`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data,
      selected_keys,
      quality_issues,
      recommendations,
    }),
  });
  if (!r.ok) throw new Error('Implement failed');
  return r.json();
}

export async function getDeploymentPlan(
  quality_issues: Record<string, unknown>,
  bias_summary: Record<string, unknown>,
  user_context: Record<string, unknown>
) {
  const r = await fetch(`${API_BASE}/api/deployment-plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      quality_issues,
      bias_summary,
      user_context,
    }),
  });
  if (!r.ok) throw new Error('Deployment plan failed');
  return r.json();
}

export async function askAI(
  question: string,
  user_context: Record<string, unknown>,
  quality_summary: Record<string, unknown>,
  bias_summary: Record<string, unknown>
): Promise<{ answer: string | null; error?: string }> {
  const r = await fetch(`${API_BASE}/api/ask-ai`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      user_context,
      quality_summary,
      bias_summary,
    }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(data.detail || 'AI chat failed');
  }
  return data;
}
