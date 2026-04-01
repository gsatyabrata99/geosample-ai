import { useState } from "react";
import { MapPin, TrendingUp, Layers } from "lucide-react";

const SCORE_COLOR = (score) => {
  if (score >= 0.85) return "#1D9E75";
  if (score >= 0.70) return "#EF9F27";
  return "#D85A30";
};

const SCORE_LABEL = (score) => {
  if (score >= 0.85) return "High";
  if (score >= 0.70) return "Moderate";
  return "Low";
};

export default function SiteMap({ sites }) {
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState(0.0);

  const filtered = sites.filter(s => s.score >= filter);

  return (
    <div className="site-map">
      <div className="map-sidebar">
        <div className="sidebar-header">
          <TrendingUp size={16} />
          <span>Ranked Sites</span>
          <span className="count">{filtered.length}</span>
        </div>

        <div className="filter-row">
          <label>Min score: <strong>{filter.toFixed(2)}</strong></label>
          <input
            type="range" min="0" max="1" step="0.05"
            value={filter}
            onChange={e => setFilter(parseFloat(e.target.value))}
          />
        </div>

        <div className="site-list">
          {filtered
            .sort((a, b) => b.score - a.score)
            .map(site => (
              <div
                key={site.site_id}
                className={`site-card ${selected?.site_id === site.site_id ? "selected" : ""}`}
                onClick={() => setSelected(site)}
              >
                <div className="site-card-header">
                  <span className="site-id">{site.site_id}</span>
                  <span
                    className="score-pill"
                    style={{ background: SCORE_COLOR(site.score) }}
                  >
                    {(site.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="site-deposit">{site.deposit}</div>
                <div className="site-meta">
                  <span>{site.grade_pct}% Cu</span>
                  <span className="viability-label">{SCORE_LABEL(site.score)}</span>
                </div>
              </div>
            ))}
        </div>
      </div>

      <div className="map-area">
        {/* Map placeholder — replace with Mapbox once token is configured */}
        <div className="map-placeholder">
          <div className="map-overlay">
            <Layers size={40} opacity={0.3} />
            <p>Central African Copperbelt</p>
            <p className="map-sub">DRC · Zambia · {filtered.length} sites plotted</p>
          </div>

          {/* SVG pseudo-map showing site positions */}
          <svg viewBox="0 0 600 400" className="pseudo-map">
            <rect width="600" height="400" fill="#1a2332" />
            {/* Grid lines */}
            {[0,1,2,3,4,5].map(i => (
              <line key={`h${i}`} x1="0" y1={i*80} x2="600" y2={i*80}
                stroke="#ffffff10" strokeWidth="1" />
            ))}
            {[0,1,2,3,4,5,6,7].map(i => (
              <line key={`v${i}`} x1={i*100} y1="0" x2={i*100} y2="400"
                stroke="#ffffff10" strokeWidth="1" />
            ))}

            {/* Site markers — positions derived from lat/lon */}
            {filtered.map((site, i) => {
              const x = ((site.lon - 24) / 6) * 500 + 50;
              const y = ((site.lat + 8) / 8) * -300 + 350;
              const r = 8 + site.score * 12;
              return (
                <g key={site.site_id} onClick={() => setSelected(site)}
                   style={{ cursor: "pointer" }}>
                  <circle cx={x} cy={y} r={r + 4}
                    fill={SCORE_COLOR(site.score)} opacity={0.2} />
                  <circle cx={x} cy={y} r={r}
                    fill={SCORE_COLOR(site.score)} opacity={0.9} />
                  <text x={x} y={y + 1} textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="8" fill="white" fontWeight="bold">
                    {(site.score * 100).toFixed(0)}
                  </text>
                </g>
              );
            })}

            {/* Labels */}
            <text x="30" y="20" fill="#ffffff60" fontSize="10">DRC</text>
            <text x="480" y="380" fill="#ffffff60" fontSize="10">Zambia</text>
          </svg>
        </div>

        {selected && (
          <div className="site-detail">
            <div className="detail-header">
              <strong>{selected.site_id}</strong>
              <span
                className="score-pill large"
                style={{ background: SCORE_COLOR(selected.score) }}
              >
                {(selected.score * 100).toFixed(0)}% Viable
              </span>
            </div>
            <div className="detail-body">
              <div className="detail-row">
                <span>Deposit</span>
                <strong>{selected.deposit}</strong>
              </div>
              <div className="detail-row">
                <span>Cu Grade</span>
                <strong>{selected.grade_pct}%</strong>
              </div>
              <div className="detail-row">
                <span>Coordinates</span>
                <strong>{selected.lat.toFixed(3)}, {selected.lon.toFixed(3)}</strong>
              </div>
              <div className="detail-row">
                <span>Viability</span>
                <strong style={{ color: SCORE_COLOR(selected.score) }}>
                  {SCORE_LABEL(selected.score)}
                </strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
