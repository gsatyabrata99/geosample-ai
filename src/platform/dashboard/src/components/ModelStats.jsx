import { Cpu, Tag, BarChart2, CheckCircle, XCircle } from "lucide-react";

export default function ModelStats({ modelInfo, health }) {
  if (!modelInfo) return <div className="empty-state">Model info unavailable</div>;

  const topFeatures = modelInfo.top_features?.slice(0, 10) || [];
  const maxImportance = Math.max(...topFeatures.map(f => f.importance), 1);

  return (
    <div className="model-stats">
      <div className="stats-grid">

        <div className="stat-card">
          <div className="stat-card-header">
            <CheckCircle size={15} />
            <span>Model Status</span>
          </div>
          {health && Object.entries(health.models).map(([k, v]) => (
            <div key={k} className="status-row">
              <span className="model-name">{k.toUpperCase()}</span>
              <span className={`status-dot ${v ? "ok" : "off"}`}>
                {v ? "● Loaded" : "○ Not loaded"}
              </span>
            </div>
          ))}
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <Tag size={15} />
            <span>NER Entity Types</span>
          </div>
          <div className="label-grid">
            {modelInfo.ner_labels?.map(label => (
              <span key={label} className="label-chip">{label}</span>
            ))}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <Cpu size={15} />
            <span>Pipeline Summary</span>
          </div>
          <div className="pipeline-rows">
            <div className="pipeline-row">
              <span>NER labels</span>
              <strong>{modelInfo.ner_labels?.length}</strong>
            </div>
            <div className="pipeline-row">
              <span>LDA topics</span>
              <strong>{modelInfo.lda_num_topics}</strong>
            </div>
            <div className="pipeline-row">
              <span>GBM features</span>
              <strong>{modelInfo.gbm_features?.length}</strong>
            </div>
            <div className="pipeline-row">
              <span>Data source</span>
              <strong>Kamoa-Kakula 2026 MRE</strong>
            </div>
            <div className="pipeline-row">
              <span>Report pages</span>
              <strong>608</strong>
            </div>
          </div>
        </div>

        <div className="stat-card wide">
          <div className="stat-card-header">
            <BarChart2 size={15} />
            <span>Top Feature Importances (GBM)</span>
          </div>
          {topFeatures.map(f => (
            <div key={f.feature} className="importance-row">
              <span className="feat-name">{f.feature.replace(/_/g, " ")}</span>
              <div className="importance-bar-bg">
                <div className="importance-bar-fill"
                  style={{ width: `${(f.importance / maxImportance) * 100}%` }} />
              </div>
              <span className="feat-val">{f.importance.toFixed(0)}</span>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
