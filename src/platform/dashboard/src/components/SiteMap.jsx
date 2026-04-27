import { useState } from "react";
import Map, { Marker, Popup, NavigationControl } from "react-map-gl";
import { TrendingUp, Layers } from "lucide-react";
import "mapbox-gl/dist/mapbox-gl.css";
import AlterationLayer from "./AlterationLayer";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

const SCORE_COLOR = (score) => {
  if (score >= 0.70) return "#1D9E75";
  if (score >= 0.40) return "#EF9F27";
  return "#D85A30";
};

const SCORE_LABEL = (score) => {
  if (score >= 0.70) return "High";
  if (score >= 0.40) return "Medium";
  return "Low";
};

export default function SiteMap({ sites }) {
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState(0.0);
  const [showAlteration, setShowAlteration] = useState(true);

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

        <div className="filter-row">
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={showAlteration}
              onChange={e => setShowAlteration(e.target.checked)}
              style={{ accentColor: "#EF9F27" }}
            />
            <Layers size={13} />
            <span>Alteration heatmap</span>
          </label>
        </div>

        <div className="site-list">
          {[...filtered]
            .sort((a, b) => b.score - a.score)
            .map(site => (
              <div
                key={site.site_id}
                className={`site-card ${selected?.site_id === site.site_id ? "selected" : ""}`}
                onClick={() => setSelected(site)}
              >
                <div className="site-card-header">
                  <span className="site-id">{site.site_id}</span>
                  <span className="score-pill" style={{ background: SCORE_COLOR(site.score) }}>
                    {(site.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="site-deposit">{site.deposit}</div>
                <div className="site-meta">
                  <span>{site.commodity}</span>
                  <span className="viability-label">{SCORE_LABEL(site.score)}</span>
                </div>
              </div>
            ))}
        </div>

        <div className="legend">
          <div className="legend-title"><Layers size={12} /> Viability Score</div>
          <div className="legend-bar">
            <div style={{height:"100%", borderRadius:"4px",
              background:"linear-gradient(to right, #D85A30, #EF9F27, #1D9E75)"}} />
          </div>
          <div className="legend-labels">
            <span>Low</span><span>High</span>
          </div>
        </div>
      </div>

      <div className="map-area">
        <Map
          initialViewState={{ longitude: 0, latitude: 20, zoom: 1.5 }}
          style={{ width: "100%", height: "100%" }}
          mapStyle="mapbox://styles/mapbox/dark-v11"
          mapboxAccessToken={MAPBOX_TOKEN}
        >
          <NavigationControl position="top-right" />
          <AlterationLayer visible={showAlteration} />

          {filtered.map(site => (
            <Marker
              key={site.site_id}
              longitude={site.lon}
              latitude={site.lat}
              anchor="center"
              onClick={e => { e.originalEvent.stopPropagation(); setSelected(site); }}
            >
              <div style={{
                width: 36 + site.score * 16,
                height: 36 + site.score * 16,
                borderRadius: "50%",
                background: SCORE_COLOR(site.score),
                opacity: 0.9,
                border: "2px solid white",
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer", color: "white", fontWeight: 700, fontSize: 11,
                boxShadow: `0 0 12px ${SCORE_COLOR(site.score)}88`,
              }}>
                {(site.score * 100).toFixed(0)}
              </div>
            </Marker>
          ))}

          {selected && (
            <Popup
              longitude={selected.lon} latitude={selected.lat}
              anchor="bottom" onClose={() => setSelected(null)}
              closeButton={true} style={{ color: "#0d1117" }}
            >
              <div style={{ padding: "4px 8px", minWidth: 200 }}>
                <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 14 }}>
                  {selected.deposit}
                  <span style={{
                    marginLeft: 8, padding: "2px 8px", borderRadius: 10,
                    background: SCORE_COLOR(selected.score),
                    color: "white", fontSize: 11, fontWeight: 700,
                  }}>
                    {(selected.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ fontSize: 12, color: "#444", marginBottom: 4 }}>
                  {selected.commodity} · {selected.country}
                </div>
                {selected.grade_pct > 0 && (
                  <div style={{ fontSize: 12 }}>Grade: <strong>{selected.grade_pct}% Cu</strong></div>
                )}
                <div style={{ fontSize: 12 }}>Viability: <strong style={{ color: SCORE_COLOR(selected.score) }}>{SCORE_LABEL(selected.score)}</strong></div>
                {selected.report && (
                  <div style={{ fontSize: 10, color: "#888", marginTop: 4 }}>{selected.report}</div>
                )}
                <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>
                  {selected.lat.toFixed(3)}, {selected.lon.toFixed(3)}
                </div>
              </div>
            </Popup>
          )}
        </Map>
      </div>
    </div>
  );
}
