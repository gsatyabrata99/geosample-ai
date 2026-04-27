import { useState, useRef } from "react";
import axios from "axios";
import { Upload, FileText, Loader, Tag, BarChart2, AlertCircle, CheckCircle } from "lucide-react";

const LABEL_COLORS = {
  ORE_GRADE: "#1D9E75",
  DEPOSIT: "#065A82",
  DRILL_HOLE: "#EF9F27",
  DEPTH: "#7F77DD",
  MINERAL: "#D85A30",
  TONNAGE: "#0F6E56",
  LOCATION: "#3B8BD4",
  COST: "#BA7517",
};

const SCORE_COLOR = (s) => {
  if (s >= 0.7) return "#1D9E75";
  if (s >= 0.4) return "#EF9F27";
  return "#D85A30";
};

export default function ReportUpload({ api }) {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f?.name.endsWith(".pdf")) {
      setError("Only PDF files are supported");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${api}/reports/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Analysis failed — check the API server");
    }
    setLoading(false);
  };

  return (
    <div className="upload-panel">
      <div className="upload-left">
        <div className="section-header">
          <Upload size={15} />
          <span>NI 43-101 Report Analysis</span>
        </div>

        <p className="upload-desc">
          Upload any NI 43-101 technical report PDF. The pipeline extracts text,
          runs NER and topic modeling, and returns a viability score with entity summary.
        </p>

        {/* Drop zone */}
        <div
          className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <div className="file-selected">
              <FileText size={32} color="#1D9E75" />
              <div>
                <div className="file-name">{file.name}</div>
                <div className="file-size">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
              </div>
              <CheckCircle size={20} color="#1D9E75" />
            </div>
          ) : (
            <div className="drop-prompt">
              <Upload size={32} opacity={0.4} />
              <p>Drop a PDF here or click to browse</p>
              <p className="drop-sub">NI 43-101 technical reports, drill logs, field notes</p>
            </div>
          )}
        </div>

        <button
          className="predict-btn"
          onClick={handleAnalyze}
          disabled={!file || loading}
        >
          {loading
            ? <><Loader size={15} className="spin" /> Analyzing {file?.name}...</>
            : <><FileText size={15} /> Analyze Report</>
          }
        </button>

        {loading && (
          <div className="progress-msg">
            <Loader size={14} className="spin" />
            Extracting text, running NER + LDA + GBM pipeline...
          </div>
        )}

        {error && (
          <div className="error-msg">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {/* Entity summary */}
        {result && (
          <div className="entity-summary">
            <div className="section-label">
              <Tag size={13} /> Extracted Entities
            </div>
            {Object.entries(result.entity_summary).map(([label, values]) => (
              <div key={label} className="entity-group">
                <span className="entity-label-chip" style={{
                  background: LABEL_COLORS[label] + "22",
                  borderColor: LABEL_COLORS[label] + "66",
                  color: LABEL_COLORS[label],
                }}>
                  {label}
                </span>
                <div className="entity-values">
                  {values.slice(0, 8).map((v, i) => (
                    <span key={i} className="entity-value">{v}</span>
                  ))}
                  {values.length > 8 && (
                    <span className="entity-more">+{values.length - 8} more</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="upload-right">
        {result ? (
          <>
            {result.commodity_warning && (
              <div style={{
                background: '#3d2a00', border: '1px solid #EF9F27',
                borderRadius: 8, padding: '10px 14px', marginBottom: 14,
                fontSize: 12, color: '#EF9F27', lineHeight: 1.5
              }}>
                ⚠️ <strong>{result.commodity}</strong> report detected.
                Viability score reflects geological data density, not copper-specific potential.
              </div>
            )}
            {/* Score */}
            <div className="score-card">
              <div className="score-label">Viability Score</div>
              <div className="score-value" style={{ color: SCORE_COLOR(result.viability_score) }}>
                {(result.viability_score * 100).toFixed(1)}%
              </div>
              <div className="score-bar-bg">
                <div className="score-bar-fill" style={{
                  width: `${result.viability_score * 100}%`,
                  background: SCORE_COLOR(result.viability_score),
                }} />
              </div>
              <div className="score-meta">
                <span>{result.filename}</span>
                <span>{(result.text_length / 1000).toFixed(0)}K chars extracted</span>
              </div>
            </div>

            {/* Key features */}
            {result.key_features && Object.keys(result.key_features).length > 0 && (
              <div className="features-card">
                <div className="section-label">
                  <BarChart2 size={13} /> Key Features
                </div>
                {Object.entries(result.key_features)
                  .filter(([k]) => !k.startsWith("topic_"))
                  .sort(([,a],[,b]) => b - a)
                  .slice(0, 8)
                  .map(([k, v]) => (
                    <div key={k} className="feature-row">
                      <span className="feat-name">{k.replace(/_/g, " ")}</span>
                      <span className="feat-val">
                        {typeof v === "number" ? v.toFixed(2) : v}
                      </span>
                    </div>
                  ))}
              </div>
            )}

            {/* Topics */}
            {result.top_topics?.length > 0 && (
              <div className="topics-card">
                <div className="section-label">
                  <Tag size={13} /> Top Topics
                </div>
                {result.top_topics.map(t => (
                  <div key={t.topic_id} className="topic-row">
                    <span className="topic-label">{t.label}</span>
                    <div className="topic-bar-bg">
                      <div className="topic-bar-fill"
                        style={{ width: `${t.probability * 100}%` }} />
                    </div>
                    <span className="topic-prob">{(t.probability * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">
            <FileText size={40} opacity={0.2} />
            <p>Upload a PDF report to see viability score, extracted entities, and geological topic distribution</p>
          </div>
        )}
      </div>
    </div>
  );
}
