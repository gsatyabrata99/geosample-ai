import { useState } from "react";
import axios from "axios";
import { Send, Loader, Tag, BarChart2 } from "lucide-react";

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

const SAMPLE_TEXTS = [
  "Drill hole DD1080 intersected 12.5m at 3.94% Cu from 450m depth in the Kakula deposit. Chalcopyrite and bornite are the primary ore minerals. The indicated resource is 523 million tonnes at 2.53% Cu.",
  "The Kamoa 3 deposit shows strong copper alteration with chalcopyrite mineralization at depths of 200-350m. Drill holes DD1720 and DD1724 returned grades of 2.1% TCu over 8m intercepts. Cut-off grade applied was 1.0% Cu.",
  "Geochemical sampling in the Western Foreland licence area returned anomalous copper values up to 0.8% Cu in soil samples. The target lies 15km from the Kamoa-Kakula mine. Follow-up drilling is recommended.",
];

export default function PredictionPanel({ api }) {
  const [text, setText] = useState(SAMPLE_TEXTS[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${api}/predict`, { text, source: "dashboard" });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Prediction failed");
    }
    setLoading(false);
  };

  const scoreColor = (s) => {
    if (s >= 0.7) return "#1D9E75";
    if (s >= 0.4) return "#EF9F27";
    return "#D85A30";
  };

  const highlightEntities = (text, entities) => {
    if (!entities?.length) return text;
    const parts = [];
    let last = 0;
    const sorted = [...entities].sort((a, b) => a.start - b.start);
    sorted.forEach(ent => {
      if (ent.start > last) parts.push(text.slice(last, ent.start));
      parts.push(
        <mark key={ent.start} style={{
          background: LABEL_COLORS[ent.label] + "33",
          borderBottom: `2px solid ${LABEL_COLORS[ent.label]}`,
          borderRadius: "2px", padding: "0 2px",
        }} title={ent.label}>
          {text.slice(ent.start, ent.end)}
        </mark>
      );
      last = ent.end;
    });
    if (last < text.length) parts.push(text.slice(last));
    return parts;
  };

  return (
    <div className="predict-panel">
      <div className="predict-left">
        <div className="section-header">
          <Send size={15} />
          <span>Geological Text Analysis</span>
        </div>

        <div className="sample-btns">
          {SAMPLE_TEXTS.map((t, i) => (
            <button key={i} className="sample-btn"
              onClick={() => { setText(t); setResult(null); }}>
              Sample {i + 1}
            </button>
          ))}
        </div>

        <textarea
          className="text-input"
          value={text}
          onChange={e => setText(e.target.value)}
          rows={6}
          placeholder="Paste geological text from a drill report or field note..."
        />

        <button className="predict-btn" onClick={predict} disabled={loading || !text}>
          {loading ? <Loader size={15} className="spin" /> : <Send size={15} />}
          {loading ? "Analyzing..." : "Analyze Text"}
        </button>

        {error && <div className="error-msg">{error}</div>}

        {result && (
          <div className="highlighted-text">
            <div className="section-label">Extracted entities</div>
            <p className="entity-text">{highlightEntities(text, result.entities)}</p>
            <div className="legend">
              {Object.entries(LABEL_COLORS).map(([label, color]) => (
                <span key={label} className="legend-item">
                  <span style={{ background: color + "33", borderBottom: `2px solid ${color}`, padding: "0 4px", borderRadius: "2px", fontSize: "11px" }}>
                    {label}
                  </span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="predict-right">
        {result ? (
          <>
            {result.commodity_warning && (
          <div style={{
            background: '#3d2a00', border: '1px solid #EF9F27',
            borderRadius: 8, padding: '10px 14px', marginBottom: 14,
            fontSize: 12, color: '#EF9F27', lineHeight: 1.5
          }}>
            ⚠️ <strong>{result.commodity}</strong> report detected.{' '}
            {result.commodity_warning}
          </div>
        )}
        <div className="score-card">
              <div style={{display:'flex', justifyContent:'center', gap:8, marginBottom:8}}>
                <span style={{padding:'2px 10px', borderRadius:10, fontSize:11, fontWeight:600,
                  background: result.is_copper ? '#1D9E751a' : '#EF9F271a',
                  color: result.is_copper ? '#1D9E75' : '#EF9F27',
                  border: `1px solid ${result.is_copper ? '#1D9E75' : '#EF9F27'}44`}}>
                  {result.commodity}
                </span>
                <span style={{padding:'2px 10px', borderRadius:10, fontSize:11,
                  background:'#ffffff11', color:'#7d8590'}}>
                  {(result.commodity_confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
              <div className="score-label">Viability Score</div>
              <div className="score-value" style={{ color: scoreColor(result.viability_score) }}>
                {(result.viability_score * 100).toFixed(1)}%
              </div>
              <div className="score-bar-bg">
                <div className="score-bar-fill"
                  style={{
                    width: `${result.viability_score * 100}%`,
                    background: scoreColor(result.viability_score),
                  }} />
              </div>
              <div className="score-time">{result.processing_time_ms}ms</div>
            </div>

            <div className="features-card">
              <div className="section-label">
                <BarChart2 size={13} /> Key Features
              </div>
              {Object.entries(result.features)
                .filter(([k, v]) => v > 0 && k !== "text_length")
                .sort(([,a],[,b]) => b - a)
                .slice(0, 8)
                .map(([k, v]) => (
                  <div key={k} className="feature-row">
                    <span className="feat-name">{k.replace(/_/g, " ")}</span>
                    <span className="feat-val">{typeof v === "number" ? v.toFixed(2) : v}</span>
                  </div>
                ))}
            </div>

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
          </>
        ) : (
          <div className="empty-state">
            <BarChart2 size={40} opacity={0.2} />
            <p>Run analysis to see viability score, extracted entities, and topic distribution</p>
          </div>
        )}
      </div>
    </div>
  );
}
