import { useState, useCallback, useEffect } from 'react';
import {
  listDemos,
  phiScanDemo,
  phiScanFile,
  generateSynthetic,
  analysisQuality,
  analysisBias,
  getRecommendations,
  implementChanges,
  getDeploymentPlan,
  askAI,
  type UserContext,
} from './api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

type Step = 'onboarding' | 'dataset' | 'synthetic' | 'analysis';
type TabId = 'quality' | 'bias' | 'deployment' | 'ask' | 'implement';

const TIMELINE_DAYS: Record<string, number> = {
  '<30 days (urgent)': 30,
  '30-60 days': 45,
  '60-90 days': 75,
  '90+ days (flexible)': 120,
};

const TAB_LABELS: Record<TabId, string> = {
  quality: 'Data Quality',
  bias: 'Bias & Fairness',
  deployment: 'Deployment',
  ask: 'Ask AI',
  implement: 'Implement Changes',
};

function Expander({
  title,
  defaultOpen = false,
  children,
}: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`expander ${open ? 'expanded' : ''}`}>
      <button type="button" className="expander-header" onClick={() => setOpen(!open)}>
        {title}
        <span className="expander-icon">▼</span>
      </button>
      <div className="expander-body">{children}</div>
    </div>
  );
}

export default function App() {
  const [step, setStep] = useState<Step>('onboarding');
  const [userContext, setUserContext] = useState<UserContext | null>(null);
  const [demos, setDemos] = useState<string[]>([]);
  const [demoOption, setDemoOption] = useState<string>('Upload Your Own');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [phiResult, setPhiResult] = useState<{
    is_safe: boolean;
    violations: string[];
    data?: Record<string, unknown>[];
    rows?: number;
    columns?: number;
  } | null>(null);
  const [syntheticData, setSyntheticData] = useState<Record<string, unknown>[] | null>(null);
  const [syntheticValidation, setSyntheticValidation] = useState<{ statistical_similarity?: number; privacy_safe?: boolean } | null>(null);
  const [qualityResult, setQualityResult] = useState<{ issues: Record<string, unknown>; summary: Record<string, unknown> } | null>(null);
  const [biasResult, setBiasResult] = useState<{ issues: Record<string, unknown>; summary: Record<string, unknown>; target_col?: string } | null>(null);
  const [recommendations, setRecommendations] = useState<Record<string, unknown>[]>([]);
  const [activeTab, setActiveTab] = useState<TabId>('quality');
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
  const [modifiedData, setModifiedData] = useState<Record<string, unknown>[] | null>(null);
  const [implementApplied, setImplementApplied] = useState<string[]>([]);
  const [implementBeforeSummary, setImplementBeforeSummary] = useState<{ total_rows: number; duplicate_count: number; missing_per_column: Record<string, number> } | null>(null);
  const [implementAfterSummary, setImplementAfterSummary] = useState<{ total_rows: number; duplicate_count: number; missing_per_column: Record<string, number> } | null>(null);
  const [deploymentPlan, setDeploymentPlan] = useState<Record<string, unknown> | null>(null);
  const [deploymentPlanLoading, setDeploymentPlanLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [askQuestion, setAskQuestion] = useState('');
  const [askAnswer, setAskAnswer] = useState<string | null>(null);
  const [askError, setAskError] = useState<string | null>(null);
  const [askLoading, setAskLoading] = useState(false);

  const loadDemos = useCallback(async () => {
    try {
      const { demos: d } = await listDemos();
      setDemos(d);
      setDemoOption('Upload Your Own');
    } catch {
      setDemos(['Demo 1: Heart Disease (Quality Issues)', 'Demo 2: Diabetes (Gender Bias)', 'Demo 3: Heart Disease (Indigenous Bias)', 'Demo 4: Combined Problems']);
      setDemoOption('Upload Your Own');
    }
  }, []);

  const handleOnboardingSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const project_description = (form.querySelector('[name="project_description"]') as HTMLTextAreaElement)?.value?.trim();
    if (!project_description) {
      setError('Please fill in project description');
      return;
    }
    const model_type = (form.querySelector('[name="model_type"]') as HTMLSelectElement)?.value ?? '';
    const use_case = (form.querySelector('[name="use_case"]') as HTMLSelectElement)?.value ?? '';
    const timeline = (form.querySelector('[name="timeline"]') as HTMLSelectElement)?.value ?? '';
    const can_collect_data = (form.querySelector('[name="can_collect_data"]') as HTMLSelectElement)?.value ?? '';
    const location = (form.querySelector('[name="location"]') as HTMLInputElement)?.value?.trim() ?? 'Not specified';
    setUserContext({
      project_description,
      model_type,
      use_case,
      timeline_days: TIMELINE_DAYS[timeline] ?? 90,
      can_collect_data,
      location,
    });
    setError(null);
    setStep('dataset');
    loadDemos();
  };

  const handleLoadDataset = async () => {
    setError(null);
    setLoading(true);
    try {
      if (demoOption && demoOption !== 'Upload Your Own' && !uploadFile) {
        const res = await phiScanDemo(demoOption);
        setPhiResult({
          is_safe: res.is_safe,
          violations: res.violations || [],
          data: res.data,
          rows: res.rows,
          columns: res.columns,
        });
      } else if (uploadFile) {
        const res = await phiScanFile(uploadFile);
        const text = await uploadFile.text();
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        const rows: Record<string, unknown>[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',');
          const row: Record<string, unknown> = {};
          headers.forEach((h, j) => { row[h.trim()] = values[j]?.trim() ?? ''; });
          rows.push(row);
        }
        setPhiResult({
          is_safe: res.is_safe,
          violations: res.violations || [],
          data: rows,
          rows: rows.length,
          columns: headers.length,
        });
      } else {
        setError('Select a demo or upload a CSV');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  const handlePhiContinue = () => {
    if (phiResult?.is_safe) setStep('synthetic');
  };

  const handleGenerateSynthetic = async () => {
    if (!phiResult?.data) return;
    setError(null);
    setLoading(true);
    try {
      const n = phiResult.data.length;
      const res = await generateSynthetic({
        demo_name: uploadFile ? undefined : demoOption || undefined,
        file: uploadFile || undefined,
        n_samples: n,
      });
      setSyntheticData(res.synthetic_data);
      setSyntheticValidation(res.validation);
      setStep('analysis');
      const q = await analysisQuality(res.synthetic_data);
      setQualityResult(q);
      const b = await analysisBias(res.synthetic_data);
      setBiasResult(b);
      const recs = await getRecommendations(q.issues, b.issues);
      setRecommendations(recs.recommendations || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed');
    } finally {
      setLoading(false);
    }
  };

  const toggleRecommendation = (key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const handleImplement = async () => {
    if (!syntheticData || selectedKeys.size === 0 || !qualityResult) return;
    setError(null);
    setLoading(true);
    try {
      const res = await implementChanges(
        syntheticData,
        Array.from(selectedKeys),
        qualityResult.issues,
        recommendations
      );
      setModifiedData(res.data);
      setImplementApplied(res.applied || []);
      setImplementBeforeSummary(res.before_summary ?? null);
      setImplementAfterSummary(res.after_summary ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Implement failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = (data: Record<string, unknown>[], filename: string) => {
    if (!data.length) return;
    const headers = Object.keys(data[0]);
    const line = (row: Record<string, unknown>) => headers.map((h) => JSON.stringify(row[h] ?? '')).join(',');
    const csv = [headers.join(','), ...data.map(line)].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleAskAI = async () => {
    if (!askQuestion.trim() || !userContext || !qualityResult || !biasResult) return;
    setAskLoading(true);
    setAskAnswer(null);
    setAskError(null);
    try {
      const res = await askAI(
        askQuestion.trim(),
        userContext,
        (qualityResult.summary ?? {}) as Record<string, unknown>,
        (biasResult.summary ?? {}) as Record<string, unknown>
      );
      if (res.error) {
        setAskError(res.error);
        setAskAnswer(null);
      } else {
        setAskAnswer(res.answer ?? null);
        setAskError(null);
      }
    } catch (e) {
      setAskError(e instanceof Error ? e.message : 'AI chat failed');
      setAskAnswer(null);
    } finally {
      setAskLoading(false);
    }
  };

  const handleReset = () => {
    setStep('onboarding');
    setUserContext(null);
    setUploadFile(null);
    setPhiResult(null);
    setSyntheticData(null);
    setSyntheticValidation(null);
    setQualityResult(null);
    setBiasResult(null);
    setRecommendations([]);
    setActiveTab('quality');
    setSelectedKeys(new Set());
    setModifiedData(null);
    setImplementApplied([]);
    setDeploymentPlan(null);
    setError(null);
    setImplementBeforeSummary(null);
    setImplementAfterSummary(null);
  };

  // Fetch deployment plan when user opens Deployment tab
  useEffect(() => {
    if (activeTab !== 'deployment' || deploymentPlan || deploymentPlanLoading || !qualityResult || !biasResult || !userContext) return;
    setDeploymentPlanLoading(true);
    setError(null);
    getDeploymentPlan(qualityResult.issues, biasResult.summary, userContext)
      .then((plan) => setDeploymentPlan(plan))
      .catch((err) => setError(err instanceof Error ? err.message : 'Deployment plan failed'))
      .finally(() => setDeploymentPlanLoading(false));
  }, [activeTab, qualityResult, biasResult, userContext, deploymentPlan, deploymentPlanLoading]);

  useEffect(() => {
    loadDemos();
  }, [loadDemos]);

  const renderRightContent = () => {
    if (step !== 'analysis') return null;
    if (activeTab === 'quality' && qualityResult) {
        const q = qualityResult;
        const missing = (q.issues.missing_values as Record<string, { count: number; percentage: number; total_affected_rows?: number; recommendations?: { method: string; priority: string; reason: string; code?: string; impact?: string }[] }>) || {};
        const dup = q.issues.duplicates as { count?: number; recommendation?: { method: string; priority: string; reason: string; code?: string; impact?: string } } | undefined;
        const outliers = (q.issues.outliers as Record<string, { count: number; bounds?: { lower: number; upper: number }; values_sample?: unknown; recommendations?: { method: string; priority: string; reason: string; code?: string }[] }>) || {};
        return (
          <div className="right-content">
            <h2 className="right-title">Data Quality</h2>
            <p className="right-subtitle">Missing values, duplicates, and outliers with recommendations.</p>
            <div className="section">
              <div className="section-title">⚠️ Missing Values</div>
              {Object.keys(missing).length > 0 ? (
                Object.entries(missing).map(([col, info]) => (
                  <Expander key={col} title={`❌ ${col}: ${info.count} missing (${info.percentage}%)`} defaultOpen={false}>
                    <p><strong>Affected rows:</strong> {info.total_affected_rows ?? '—'}</p>
                    <div className="section-title">🤖 Recommendations</div>
                    {info.recommendations?.slice(0, 3).map((rec, i) => (
                      <div key={i} className="card" style={{ marginBottom: 12 }}>
                        <strong>{rec.method}</strong> {rec.priority}
                        <p style={{ margin: '4px 0', color: 'var(--text-secondary)' }}>💡 {rec.reason}</p>
                        {rec.code && <pre className="code-block">{rec.code}</pre>}
                        {rec.impact && <p style={{ margin: '4px 0', fontSize: '0.9rem', color: 'var(--text-muted)' }}>✨ Impact: {rec.impact}</p>}
                      </div>
                    ))}
                  </Expander>
                ))
              ) : (
                <div className="alert alert-success">✅ No missing values detected!</div>
              )}
            </div>
            <div className="section">
              <div className="section-title">🔄 Duplicate Records</div>
              {dup && dup.count > 0 ? (
                <Expander title={`⚠️ Found ${dup.count} duplicate records`} defaultOpen={false}>
                  {dup.recommendation && (
                    <>
                      <p><strong>{dup.recommendation.priority}</strong> {dup.recommendation.method}</p>
                      <p style={{ color: 'var(--text-secondary)' }}>💡 {dup.recommendation.reason}</p>
                      {dup.recommendation.code && <pre className="code-block">{dup.recommendation.code}</pre>}
                      {dup.recommendation.impact && <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>✨ {dup.recommendation.impact}</p>}
                    </>
                  )}
                </Expander>
              ) : (
                <div className="alert alert-success">✅ No duplicates detected!</div>
              )}
            </div>
            <div className="section">
              <div className="section-title">📈 Outlier Detection</div>
              {Object.keys(outliers).length > 0 ? (
                Object.entries(outliers).map(([col, info]) => (
                  <Expander key={col} title={`⚡ ${col}: ${info.count} outliers detected`} defaultOpen={false}>
                    {info.bounds && <p><strong>Expected range:</strong> {info.bounds.lower.toFixed(1)} – {info.bounds.upper.toFixed(1)}</p>}
                    {info.values_sample != null && <p><strong>Sample values:</strong> {String(info.values_sample)}</p>}
                    <div className="section-title">Recommendations</div>
                    {info.recommendations?.map((r, i) => (
                      <div key={i} style={{ marginBottom: 8 }}><strong>{r.method}</strong> {r.priority} — {r.reason} {r.code && <pre className="code-block">{r.code}</pre>}</div>
                    ))}
                  </Expander>
                ))
              ) : (
                <div className="alert alert-success">✅ No significant outliers detected!</div>
              )}
            </div>
          </div>
        );
      }
      if (activeTab === 'bias' && biasResult) {
        type BiasInfo = { distribution?: Record<string, number>; bias_detected?: boolean; issues?: string[] };
        return (
          <div className="right-content">
            <h2 className="right-title">Bias & Fairness Analysis</h2>
            <p className="right-subtitle">Demographic balance and fairness metrics. Target: &lt;5% gap for binary attributes (e.g. sex/gender).</p>
            <div className="metrics-row">
              <div className="metric">
                <div className="metric-value">{String(biasResult.summary?.attributes_analyzed ?? 0)}</div>
                <div className="metric-label">Attributes analyzed</div>
              </div>
              <div className="metric">
                <div className="metric-value">{String(biasResult.summary?.attributes_with_bias ?? 0)}</div>
                <div className="metric-label">With bias</div>
              </div>
              <div className="metric">
                <div className="metric-value">{String(biasResult.summary?.status ?? '—')}</div>
                <div className="metric-label">Overall status</div>
              </div>
            </div>
            {Object.entries(biasResult.issues).map(([col, infoRaw]) => {
              const info = infoRaw as BiasInfo;
              const dist = info.distribution ?? {};
              const distEntries = Object.entries(dist).map(([name, value]) => ({ name, value: Number(value) }));
              const values = distEntries.map((d) => d.value);
              const maxPct = values.length ? Math.max(...values) : 0;
              const minPct = values.length ? Math.min(...values) : 0;
              const gapPct = maxPct - minPct;
              const threshold = 5;
              const isBinary = values.length === 2;
              const biasDetected = info.bias_detected ?? gapPct > threshold;
              const issuesList = info.issues ?? [];
              return (
                <div key={col} className="bias-attribute-block">
                  <div className="bias-attribute-title">{col}</div>
                  <div className="bias-stats-row">
                    <span className="bias-stat"><strong>Distribution gap:</strong> {gapPct.toFixed(1)}%</span>
                    <span className="bias-stat"><strong>Fairness threshold:</strong> {isBinary ? `\u003c${threshold}% for binary` : '\u003c20% per group'}</span>
                    <span className={`bias-badge ${biasDetected ? 'detected' : 'balanced'}`}>
                      {biasDetected ? 'Bias detected' : 'Balanced'}
                    </span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Group</th>
                        <th>Percentage</th>
                        <th>Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {distEntries.map(({ name, value }) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td><strong>{value.toFixed(1)}%</strong></td>
                          <td>
                            {value < 20 ? 'Underrepresented' : value >= 45 && value <= 55 ? 'Balanced' : value > 55 ? 'Overrepresented' : 'Slight imbalance'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {issuesList.length > 0 && (
                    <>
                      <div className="section-title">Why bias was flagged</div>
                      <ul className="bias-issues-list">
                        {issuesList.map((issue, i) => (
                          <li key={i}>{issue}</li>
                        ))}
                      </ul>
                    </>
                  )}
                  <div className="bias-chart-wrap">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={distEntries} layout="vertical" margin={{ left: 60 }}>
                        <XAxis type="number" domain={[0, 100]} />
                        <YAxis type="category" dataKey="name" width={80} />
                        <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, 'Share']} />
                        <Bar dataKey="value" name="Share (%)">
                          {distEntries.map((d, i) => (
                            <Cell
                              key={i}
                              fill={d.value < 20 ? '#dc2626' : d.value >= 45 && d.value <= 55 ? '#059669' : '#d97706'}
                            />
                          ))}
                        </Bar>
                        <ReferenceLine x={50} stroke="#94a3b8" strokeDasharray="4 4" />
                      </BarChart>
                    </ResponsiveContainer>
                    <p className="bias-target-note">Dashed line: 50% (balanced). Green: 45–55%; Amber: outside range; Red: under 20%.</p>
                  </div>
                </div>
              );
            })}
          </div>
        );
      }
      if (activeTab === 'deployment') {
        if (deploymentPlanLoading) {
          return (
            <div className="empty-state">
              <p>Generating week-by-week deployment plan…</p>
            </div>
          );
        }
        if (deploymentPlan) {
          const plan = deploymentPlan as {
            recommended_strategy?: string;
            why_this_strategy?: string;
            weekly_plan?: { week_number: number; days_range: string; tasks: string[]; deliverables: string[]; estimated_hours?: number }[] | Record<string, { week_number: number; days_range: string; tasks: string[]; deliverables: string[]; estimated_hours?: number }>;
            critical_path_items?: string[];
            risk_factors?: string[];
            alternative_paths?: { scenario: string; when_to_pivot: string; modified_timeline: string }[];
            success_metrics?: string[];
          };
          const weeklyPlan = plan.weekly_plan == null ? [] : Array.isArray(plan.weekly_plan) ? plan.weekly_plan : Object.values(plan.weekly_plan);
          return (
            <div className="right-content">
              <h2 className="right-title">Deployment roadmap</h2>
              <p className="right-subtitle">Week-by-week plan from your quality and bias summary.</p>
              <div className="alert alert-info" style={{ marginBottom: 24 }}>
                <strong>Recommended strategy</strong>
                <p style={{ margin: '8px 0 0' }}>{plan.recommended_strategy ?? '—'}</p>
                <p style={{ margin: '8px 0 0', opacity: 0.9 }}><strong>Rationale:</strong> {plan.why_this_strategy ?? '—'}</p>
              </div>
              <div className="section-title">Week-by-week implementation</div>
              {weeklyPlan.map((week) => (
                <div key={week.week_number} className="week-card">
                  <div className="week-card-title">Week {week.week_number}: {week.days_range}</div>
                  <ul>
                    {(week.tasks ?? []).map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                  <p style={{ margin: '8px 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}><strong>Deliverables:</strong> {(week.deliverables ?? []).join(', ')}</p>
                  {week.estimated_hours != null && <div className="week-card-meta">Est. {week.estimated_hours} hours</div>}
                </div>
              ))}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginTop: 24 }}>
                <div>
                  <div className="section-title">Critical path</div>
                  <ul className="list-plain">{(plan.critical_path_items ?? []).map((item, i) => <li key={i}>{item}</li>)}</ul>
                  <div className="section-title" style={{ marginTop: 16 }}>Risk factors</div>
                  <ul className="list-plain">{(plan.risk_factors ?? []).map((r, i) => <li key={i}>{r}</li>)}</ul>
                </div>
                <div>
                  <div className="section-title">Success metrics</div>
                  <ul className="list-plain">{(plan.success_metrics ?? []).map((m, i) => <li key={i}>{m}</li>)}</ul>
                  {(plan.alternative_paths ?? []).length > 0 && (
                    <>
                      <div className="section-title" style={{ marginTop: 16 }}>Alternative paths</div>
                      {(plan.alternative_paths ?? []).map((alt, i) => (
                        <div key={i} className="card" style={{ marginBottom: 8 }}>
                          <strong>{alt.scenario}</strong> — {alt.when_to_pivot}. {alt.modified_timeline}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        }
        return (
          <div className="empty-state">
            <p>Select the Deployment tab after analysis. The plan will load here (uses Ollama if available).</p>
          </div>
        );
      }
      if (activeTab === 'ask') {
        if (!userContext || !qualityResult || !biasResult) {
          return (
            <div className="right-content">
              <h2 className="right-title">Ask AI</h2>
              <p className="right-subtitle">Run the full analysis first, then you can ask questions about the findings.</p>
              <div className="empty-state">
                <p>Complete Steps 1–3 and run analysis. The AI will then have project context, quality summary, and bias summary to answer your questions.</p>
              </div>
            </div>
          );
        }
        return (
          <div className="right-content">
            <h2 className="right-title">Ask AI</h2>
            <p className="right-subtitle">Ask free-text questions about your project, quality issues, or bias results.</p>
            <div className="card">
              <div className="section-title">Question</div>
              <textarea
                className="form-input form-textarea"
                placeholder="Example: Explain why the gender distribution is biased and how that affects my model."
                rows={3}
                value={askQuestion}
                onChange={(e) => setAskQuestion(e.target.value)}
              />
              <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={handleAskAI}
                  disabled={askLoading || !askQuestion.trim()}
                >
                  {askLoading ? 'Asking…' : 'Send'}
                </button>
              </div>
              {askAnswer && (
                <div style={{ marginTop: 16 }}>
                  <div className="section-title">AI response</div>
                  <div className="card" style={{ marginTop: 8 }}>
                    <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{askAnswer}</p>
                  </div>
                </div>
              )}
              {askError && (
                <div className="alert alert-error" style={{ marginTop: 16 }}>
                  {askError}
                </div>
              )}
            </div>
          </div>
        );
      }
      if (activeTab === 'implement') {
        const typeLabels: Record<string, string> = {
          missing_value: 'Missing values',
          duplicate: 'Duplicates',
          outlier: 'Outliers',
          bias_normalization: 'Bias normalization',
          bias_mitigation: 'Bias mitigation (SMOTE)',
          other: 'Other',
        };
        const getType = (rec: Record<string, unknown>) => {
          const key = String(rec.key ?? '');
          if (key.startsWith('missing_')) return 'missing_value';
          if (key.startsWith('duplicate')) return 'duplicate';
          if (key.startsWith('outlier_')) return 'outlier';
          if (key.startsWith('bias_normalize')) return 'bias_normalization';
          if (key.startsWith('bias_mitigation')) return 'bias_mitigation';
          return 'other';
        };
        const grouped = recommendations.reduce<Record<string, Record<string, unknown>[]>>((acc, rec) => {
          const t = getType(rec);
          if (!acc[t]) acc[t] = [];
          acc[t].push(rec);
          return acc;
        }, {});
        return (
          <div className="right-content">
            <h2 className="right-title">Implement changes</h2>
            <p className="right-subtitle">Select recommendations to apply, then download the improved dataset.</p>
            {recommendations.length === 0 ? (
              <div className="alert alert-success">✅ No recommendations available! Your dataset appears to be clean.</div>
            ) : (
              <>
              <div className="alert alert-info">{recommendations.length} recommendation(s) available.</div>
                <div className="section-title">Select Changes to Implement</div>
                {Object.entries(grouped).map(([type, recs]) => (
                  <Expander
                    key={type}
                    title={`${typeLabels[type] ?? type} (${recs.length})`}
                    defaultOpen={true}
                  >
                    {recs.map((rec) => (
                      <div key={String(rec.key)} className="card" style={{ marginBottom: 8 }}>
                        <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', fontSize: '0.9rem' }}>
                          <input
                            type="checkbox"
                            checked={selectedKeys.has(String(rec.key))}
                            onChange={() => toggleRecommendation(String(rec.key))}
                          />
                          <span>
                            <strong>{String(rec.method)}</strong> {String(rec.priority)}<br />
                            {rec.column && <small style={{ color: 'var(--text-muted)' }}>Column: {String(rec.column)}</small>}<br />
                            💡 {String(rec.reason).slice(0, 120)}{String(rec.reason).length > 120 ? '…' : ''}
                            {rec.impact && <><br /><small>✨ Impact: {String(rec.impact)}</small></>}
                          </span>
                        </label>
                      </div>
                    ))}
                  </Expander>
                ))}
                <div style={{ marginTop: 24 }}>
                  {selectedKeys.size > 0 && <p className="alert alert-info">✅ {selectedKeys.size} change(s) selected for implementation</p>}
                  {selectedKeys.size === 0 && <p className="alert alert-error">⚠️ Please select at least one change to implement</p>}
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
                    <button
                      onClick={handleImplement}
                      disabled={loading || selectedKeys.size === 0}
                      className="btn btn-primary"
                    >
                      {loading ? 'Applying…' : 'Implement selected'}
                    </button>
                    {modifiedData && (
                      <button onClick={() => downloadCsv(modifiedData, 'improved_dataset.csv')} className="btn btn-success">
                        Download modified CSV
                      </button>
                    )}
                  </div>
                </div>
                {implementApplied.length > 0 && (
                  <div className="card" style={{ marginTop: 24 }}>
                    <div className="section-title">Applied</div>
                    <ul className="list-plain" style={{ color: 'var(--success)' }}>
                      {implementApplied.map((a, i) => <li key={i}>{a}</li>)}
                    </ul>
                  </div>
                )}
                {implementBeforeSummary && implementAfterSummary && (
                  <div className="card" style={{ marginTop: 24 }}>
                    <div className="section-title">Before vs after</div>
                    <p className="right-subtitle">Summary of how the dataset changed after applying your selected recommendations.</p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                      <div>
                        <strong>Before</strong>
                        <ul className="list-plain" style={{ marginTop: 8, fontSize: '0.9rem' }}>
                          <li>Rows: {implementBeforeSummary.total_rows}</li>
                          <li>Duplicates: {implementBeforeSummary.duplicate_count}</li>
                          {Object.entries(implementBeforeSummary.missing_per_column).length > 0 && (
                            <li>Missing: {Object.entries(implementBeforeSummary.missing_per_column).map(([col, n]) => `${col} (${n})`).join(', ')}</li>
                          )}
                          {Object.entries(implementBeforeSummary.missing_per_column).length === 0 && implementBeforeSummary.duplicate_count === 0 && (
                            <li>No missing values or duplicates</li>
                          )}
                        </ul>
                      </div>
                      <div>
                        <strong>After</strong>
                        <ul className="list-plain" style={{ marginTop: 8, fontSize: '0.9rem', color: 'var(--success)' }}>
                          <li>Rows: {implementAfterSummary.total_rows}</li>
                          <li>Duplicates: {implementAfterSummary.duplicate_count}</li>
                          {Object.entries(implementAfterSummary.missing_per_column).length > 0 ? (
                            <li>Missing: {Object.entries(implementAfterSummary.missing_per_column).map(([col, n]) => `${col} (${n})`).join(', ')}</li>
                          ) : (
                            <li>No missing values</li>
                          )}
                        </ul>
                      </div>
                    </div>
                    {syntheticData && modifiedData && (
                      <div style={{ marginTop: 16 }}>
                        <div className="section-title">Class balance for SMOTE / oversampling</div>
                        {Array.from(
                          new Set(
                            implementApplied
                              .map((txt) => {
                                const m = txt.match(/SMOTE on (.+)$/) || txt.match(/Oversampling on (.+)$/) || txt.match(/Undersampling on (.+)$/);
                                return m ? m[1] : null;
                              })
                              .filter((x): x is string => !!x)
                          )
                        ).map((col) => {
                          const buildDist = (rows: Record<string, unknown>[]) => {
                            const counts: Record<string, number> = {};
                            rows.forEach((r) => {
                              const v = String((r as Record<string, unknown>)[col] ?? '');
                              counts[v] = (counts[v] || 0) + 1;
                            });
                            const total = rows.length || 1;
                            return Object.entries(counts).map(([g, c]) => ({
                              group: g,
                              beforePct: (c / total) * 100,
                            }));
                          };
                          const beforeCounts: Record<string, number> = {};
                          syntheticData.forEach((r) => {
                            const v = String((r as Record<string, unknown>)[col] ?? '');
                            beforeCounts[v] = (beforeCounts[v] || 0) + 1;
                          });
                          const afterCounts: Record<string, number> = {};
                          modifiedData.forEach((r) => {
                            const v = String((r as Record<string, unknown>)[col] ?? '');
                            afterCounts[v] = (afterCounts[v] || 0) + 1;
                          });
                          const allGroups = Array.from(new Set([...Object.keys(beforeCounts), ...Object.keys(afterCounts)]));
                          const totalBefore = syntheticData.length || 1;
                          const totalAfter = modifiedData.length || 1;
                          return (
                            <div key={col} style={{ marginTop: 8 }}>
                              <p style={{ fontSize: '0.9rem', marginBottom: 4 }}><strong>{col}</strong> (class distribution before vs after)</p>
                              <table className="data-table">
                                <thead>
                                  <tr>
                                    <th>Group</th>
                                    <th>Before %</th>
                                    <th>After %</th>
                                    <th>Change</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {allGroups.map((g) => {
                                    const b = ((beforeCounts[g] || 0) / totalBefore) * 100;
                                    const a = ((afterCounts[g] || 0) / totalAfter) * 100;
                                    const delta = a - b;
                                    return (
                                      <tr key={g}>
                                        <td>{g}</td>
                                        <td>{b.toFixed(1)}%</td>
                                        <td>{a.toFixed(1)}%</td>
                                        <td>{delta >= 0 ? '+' : ''}{delta.toFixed(1)}%</td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        );
      }
    return null;
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <span className="app-header-title">MedGuard AI</span>
          <span className="app-header-subtitle">Healthcare data quality &amp; bias validator</span>
        </div>
      </header>

      {error && (
        <div className="alert alert-error" style={{ margin: '12px 24px' }}>
          {error}
        </div>
      )}

      <main className="app-main">
        {(step === 'dataset' || step === 'synthetic' || step === 'analysis') && (
          <div className="toolbar">
            <div className="form-group">
              <label className="form-label">Dataset</label>
              <select
                value={demoOption}
                onChange={(e) => { setDemoOption(e.target.value); setUploadFile(null); }}
                className="form-input form-select"
              >
                <option value="Upload Your Own">Upload your own CSV</option>
                {demos.length ? demos.map((d) => <option key={d} value={d}>{d}</option>) : (
                  <>
                    <option>Demo 1: Heart Disease (Quality Issues)</option>
                    <option>Demo 2: Diabetes (Gender Bias)</option>
                    <option>Demo 3: Heart Disease (Indigenous Bias)</option>
                    <option>Demo 4: Combined Problems</option>
                  </>
                )}
              </select>
            </div>
            <button type="button" onClick={handleReset} className="btn btn-ghost">
              Reset
            </button>
          </div>
        )}

        <div className="app-content">
            {step === 'onboarding' && (
              <div className="page-block">
                <h1 className="page-title">Step 1: Project context</h1>
                <p className="page-lead">Answer a few questions so we can tailor recommendations.</p>
                <form onSubmit={handleOnboardingSubmit} className="form-block">
                  <div className="form-row">
                    <div>
                      <div className="form-group">
                        <label className="form-label">What are you building?</label>
                        <textarea
                          name="project_description"
                          placeholder="e.g., Lung disease prediction model for hospital deployment"
                          rows={3}
                          className="form-input form-textarea"
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Model type</label>
                        <select name="model_type" className="form-input form-select">
                          {['Random Forest', 'Neural Network', 'Logistic Regression', 'XGBoost', 'Support Vector Machine', 'Ensemble', 'Other'].map((o) => (
                            <option key={o} value={o}>{o}</option>
                          ))}
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Use case</label>
                        <select name="use_case" className="form-input form-select">
                          {['Research study', 'Clinical deployment', 'FDA submission', 'Proof of concept', 'Academic project'].map((o) => (
                            <option key={o} value={o}>{o}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <div>
                      <div className="form-group">
                        <label className="form-label">Timeline</label>
                        <select name="timeline" className="form-input form-select">
                          {Object.keys(TIMELINE_DAYS).map((k) => (
                            <option key={k} value={k}>{k}</option>
                          ))}
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Can you collect more data if needed?</label>
                        <select name="can_collect_data" className="form-input form-select">
                          <option value="Yes, we have recruitment capabilities">Yes</option>
                          <option value="Maybe, but it's difficult">Maybe</option>
                          <option value="No, we must use existing data only">No</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Location (optional)</label>
                        <input
                          name="location"
                          type="text"
                          placeholder="e.g., Boston Medical Center"
                          className="form-input"
                        />
                      </div>
                    </div>
                  </div>
                  <button type="submit" className="btn btn-primary">
                    Continue
                  </button>
                </form>
              </div>
            )}

            {step === 'dataset' && (
              <div className="page-block">
                {userContext && (
                  <p className="page-lead" style={{ marginBottom: 8 }}><strong>Context:</strong> {userContext.model_type} · {userContext.timeline_days} days · {userContext.use_case}</p>
                )}
                <h1 className="page-title">Step 2: Load dataset</h1>
                <div className="form-group">
                  <label className="form-label">Upload CSV (or use demo above)</label>
                  <div className="file-input-wrap">
                    <input type="file" accept=".csv" onChange={(e) => { const f = e.target.files?.[0]; if (f) { setUploadFile(f); setDemoOption('Upload Your Own'); } }} />
                    {uploadFile && <p style={{ margin: '8px 0 0', color: 'var(--accent)' }}>{uploadFile.name}</p>}
                  </div>
                </div>
                <p className="page-lead" style={{ marginBottom: 12 }}>File must be de-identified. PHI scan runs automatically.</p>
                <div className="button-row">
                  <button onClick={handleLoadDataset} disabled={loading} className="btn btn-primary">
                    {loading ? 'Scanning…' : 'Load & run PHI scan'}
                  </button>
                  {phiResult?.is_safe && (
                    <button onClick={handlePhiContinue} className="btn btn-success">
                      Continue to synthetic data
                    </button>
                  )}
                </div>
                {phiResult && (
                  <div className="content-block">
                    {phiResult.is_safe ? (
                      <div className="alert alert-success">
                        <strong>Loaded.</strong> {phiResult.rows ?? 0} records, {phiResult.columns ?? 0} features. PHI scan passed; privacy firewall active.
                      </div>
                    ) : (
                      <div className="alert alert-error">
                        <strong>PHI detected.</strong> Remove identifiers and re-upload.
                        <ul className="list-plain" style={{ marginTop: 12 }}>{phiResult.violations.map((v, i) => <li key={i}>{v}</li>)}</ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {step === 'synthetic' && phiResult?.data && (
              <div className="page-block">
                <h1 className="page-title">Step 3: Synthetic data</h1>
                <p className="page-lead">Analysis runs on synthetic data only; no real patient data is used.</p>
                <button
                  onClick={handleGenerateSynthetic}
                  disabled={loading}
                  className="btn btn-primary"
                >
                  {loading ? 'Generating…' : 'Generate & run analysis'}
                </button>
              </div>
            )}

            {step === 'analysis' && (
              <div className="page-block">
                <h1 className="page-title">Analysis</h1>
                {syntheticValidation && (
                  <p className="page-lead">Synthetic similarity: <strong>{((syntheticValidation.statistical_similarity ?? 0) * 100).toFixed(1)}%</strong></p>
                )}
                {qualityResult && biasResult && (
                  <>
                    <div className="button-row" style={{ marginBottom: 24 }}>
                      <button
                        onClick={() => syntheticData && downloadCsv(syntheticData, 'synthetic_dataset.csv')}
                        className="btn btn-primary"
                      >
                        Download synthetic data
                      </button>
                    </div>
                    <Expander title="Preview & compare: original vs synthetic data" defaultOpen={false}>
                      <p className="right-subtitle">First 10 rows. Scroll horizontally on each table to see all columns.</p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                        <div>
                          <div className="section-title">Original (de-identified)</div>
                          {phiResult?.data && phiResult.data.length > 0 ? (
                            <div className="table-scroll-wrap">
                              <table className="data-table data-table-wide">
                                <thead>
                                  <tr>
                                    {Object.keys(phiResult.data[0] ?? {}).map((h) => (
                                      <th key={h}>{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {phiResult.data.slice(0, 10).map((row, i) => (
                                    <tr key={i}>
                                      {Object.keys(phiResult.data[0] ?? {}).map((k) => (
                                        <td key={k}>{String((row as Record<string, unknown>)[k] ?? '')}</td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Original data not available in this view.</p>
                          )}
                        </div>
                        <div>
                          <div className="section-title">Synthetic (used for analysis)</div>
                          {syntheticData && syntheticData.length > 0 ? (
                            <div className="table-scroll-wrap">
                              <table className="data-table data-table-wide">
                                <thead>
                                  <tr>
                                    {Object.keys(syntheticData[0] ?? {}).map((h) => (
                                      <th key={h}>{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {syntheticData.slice(0, 10).map((row, i) => (
                                    <tr key={i}>
                                      {Object.keys(syntheticData[0] ?? {}).map((k) => (
                                        <td key={k}>{String((row as Record<string, unknown>)[k] ?? '')}</td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </Expander>
                    <div className="metrics-row">
                      <div className="metric">
                        <div className="metric-value">{String(qualityResult.summary?.total_records ?? 0)}</div>
                        <div className="metric-label">Total records</div>
                      </div>
                      <div className="metric">
                        <div className="metric-value">{String(qualityResult.summary?.missing_value_columns ?? 0)}</div>
                        <div className="metric-label">Missing values</div>
                      </div>
                      <div className="metric">
                        <div className="metric-value">{String(qualityResult.summary?.duplicate_records ?? 0)}</div>
                        <div className="metric-label">Duplicates</div>
                      </div>
                      <div className="metric">
                        <div className="metric-value">{String(biasResult.summary?.attributes_with_bias ?? 0)}</div>
                        <div className="metric-label">Bias detected</div>
                      </div>
                      <div className="metric">
                        <div className="metric-value">
                          {(() => {
                            const total = (Number(qualityResult.summary?.missing_value_columns ?? 0) || 0) + (Number(qualityResult.summary?.duplicate_records ?? 0) || 0) + (Number(biasResult.summary?.attributes_with_bias ?? 0) || 0);
                            return total === 0 ? 'Clean' : total <= 3 ? 'Minor' : 'Critical';
                          })()}
                        </div>
                        <div className="metric-label">Status</div>
                      </div>
                    </div>
                    <div style={{ height: 1, background: 'var(--border)', margin: 'var(--space) 0' }} />
                    <nav className="tabs-nav tabs-nav-horizontal">
                      {(['quality', 'bias', 'deployment', 'ask', 'implement'] as TabId[]).map((tab) => (
                        <button
                          key={tab}
                          type="button"
                          onClick={() => setActiveTab(tab)}
                          className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                        >
                          {TAB_LABELS[tab]}
                        </button>
                      ))}
                    </nav>
                    <div className="tab-content">
                      {renderRightContent()}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </main>
    </div>
  );
}
