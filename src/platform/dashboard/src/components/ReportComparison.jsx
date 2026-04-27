import { useState, useRef } from "react";
import axios from "axios";
import { Upload, FileText, Loader, BarChart2, Plus, Trash2 } from "lucide-react";

const SCORE_COLOR = (s) => {
  if (s >= 70) return "#1D9E75";
  if (s >= 40) return "#EF9F27";
  return "#D85A30";
};

const SCORE_LABEL = (s) => {
  if (s >= 70) return "High";
  if (s >= 40) return "Medium";
  return "Low";
};

const PRESET_REPORTS = [
  { name: "Kamoa-Kakula 2026 MRE", score: 99.9, commodity: "Copper", country: "DRC", grade: "2.86% Cu" },
  { name: "Kazhiba Copper Project", score: 98.2, commodity: "Copper", country: "DRC", grade: "High grade" },
  { name: "First Quantum Cayeli", score: 99.5, commodity: "Copper", country: "Turkey", grade: "Producing" },
  { name: "Aldebaran Rio Grande", score: 96.5, commodity: "Copper-Gold", country: "Argentina", grade: "Porphyry" },
  { name: "Kenorland Frotet", score: 30.9, commodity: "Gold", country: "Canada", grade: "2.55 Moz Au" },
  { name: "Skeena Eskay Creek", score: 11.6, commodity: "Gold-Silver", country: "Canada", grade: "High grade Au" },
  { name: "Estrades PEA", score: 7.5, commodity: "Zinc-Gold", country: "Canada", grade: "Polymetallic" },
];

export default function ReportComparison({ api }) {
  const [reports, setReports] = useState(PRESET_REPORTS);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = async (file) => {
    if (!file?.name.endsWith(".pdf")) {
      setError("Only PDF files supported");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${api}/reports/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });
      const data = res.data;
      const newReport = {
        name: file.name.replace(".pdf", "").slice(0, 40),
        score: data.viability_score * 100,
        commodity: "Unknown",
        country: "Unknown",
        grade: `${(data.text_length/1000).toFixed(0)}K chars`,
        entities: data.entity_summary,
        topics: data.top_topics,
      };
      setReports(prev => [...prev, newReport].sort((a, b) => b.score - a.score));
    } catch (e) {
      setError("Analysis failed — check API is running");
    }
    setUploading(false);
  };

  const removeReport = (idx) => {
    setReports(prev => prev.filter((_, i) => i !== idx));
  };

  const sorted = [...reports].sort((a, b) => b.score - a.score);
  const maxScore = Math.max(...sorted.map(r => r.score));

  return (
    <div className="comparison-panel">
      <div className="comparison-header">
        <div>
          <h2 className="comparison-title">Report Comparison</h2>
          <p className="comparison-subtitle">
            Upload NI 43-101 reports to rank them by copper exploration viability.
            Pre-loaded with 7 benchmark reports.
          </p>
        </div>
        <div className="comparison-actions">
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            onChange={e => handleFile(e.target.files[0])}
          />
          <button
            className="add-report-btn"
            onClick={() => inputRef.current.click()}
            disabled={uploading}
          >
            {uploading
              ? <><Loader size={15} className="spin" /> Analyzing...</>
              : <><Plus size={15} /> Add Report</>
            }
          </button>
        </div>
      </div>

      {error && <div className="error-msg" style={{margin: "0 0 16px 0"}}>{error}</div>}

      {/* Bar chart */}
      <div className="comparison-chart">
        {sorted.map((report, idx) => (
          <div key={idx} className="comparison-row">
            <div className="comparison-name">
              <span className="rank-badge">#{idx + 1}</span>
              <span className="report-name-text">{report.name}</span>
              <span className="commodity-tag">{report.commodity}</span>
            </div>
            <div className="comparison-bar-container">
              <div
                className="comparison-bar"
                style={{
                  width: `${(report.score / maxScore) * 100}%`,
                  background: SCORE_COLOR(report.score),
                }}
              />
              <span className="comparison-score" style={{ color: SCORE_COLOR(report.score) }}>
                {report.score.toFixed(1)}%
              </span>
              <span className="viability-tag" style={{ color: SCORE_COLOR(report.score) }}>
                {SCORE_LABEL(report.score)}
              </span>
            </div>
            <button
              className="remove-btn"
              onClick={() => removeReport(reports.indexOf(report))}
              title="Remove"
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      {/* Summary cards */}
      <div className="comparison-cards">
        <div className="summary-card">
          <div className="summary-card-title">Highest Viability</div>
          <div className="summary-card-value" style={{ color: "#1D9E75" }}>
            {sorted[0]?.name}
          </div>
          <div className="summary-card-score" style={{ color: "#1D9E75" }}>
            {sorted[0]?.score.toFixed(1)}%
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-card-title">Reports Analyzed</div>
          <div className="summary-card-value">{reports.length}</div>
          <div className="summary-card-sub">
            {reports.filter(r => r.score >= 70).length} high viability
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-card-title">Score Range</div>
          <div className="summary-card-value">
            {Math.min(...reports.map(r => r.score)).toFixed(1)}% —{" "}
            {Math.max(...reports.map(r => r.score)).toFixed(1)}%
          </div>
          <div className="summary-card-sub">Across all reports</div>
        </div>
        <div className="summary-card">
          <div className="summary-card-title">Lowest Viability</div>
          <div className="summary-card-value" style={{ color: "#D85A30" }}>
            {sorted[sorted.length - 1]?.name}
          </div>
          <div className="summary-card-score" style={{ color: "#D85A30" }}>
            {sorted[sorted.length - 1]?.score.toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
