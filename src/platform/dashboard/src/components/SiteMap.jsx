import { useState } from "react";
import Map, { Marker, Popup, NavigationControl } from "react-map-gl";
import { TrendingUp } from "lucide-react";
import "mapbox-gl/dist/mapbox-gl.css";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

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
                  <span className="score-pill" style={{ background: SCORE_COLOR(site.score) }}>
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
        <Map
          initialViewState={{
            longitude: 26.5,
            latitude: -11.5,
            zoom: 5.5,
          }}
          style={{ width: "100%", height: "100%" }}
          mapStyle="mapbox://styles/mapbox/dark-v11"
          mapboxAccessToken={MAPBOX_TOKEN}
        >
          <NavigationControl position="top-right" />

          {filtered.map(site => (
            <Marker
              key={site.site_id}
              longitude={site.lon}
              latitude={site.lat}
              anchor="center"
              onClick={e => { e.originalEvent.stopPropagation(); setSelected(site); }}
            >
              <div style={{
                width: 40 + site.score * 20,
                height: 40 + site.score * 20,
                borderRadius: "50%",
                background: SCORE_COLOR(site.score),
                opacity: 0.85,
                border: "2px solid white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                color: "white",
                fontWeight: 700,
                fontSize: 12,
                boxShadow: `0 0 12px ${SCORE_COLOR(site.score)}88`,
              }}>
                {(site.score * 100).toFixed(0)}
              </div>
            </Marker>
          ))}

          {selected && (
            <Popup
              longitude={selected.lon}
              latitude={selected.lat}
              anchor="bottom"
              onClose={() => setSelected(null)}
              closeButton={true}
              style={{ color: "#0d1117" }}
            >
              <div style={{ padding: "4px 8px", minWidth: 180 }}>
                <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 14 }}>
                  {selected.site_id}
                  <span style={{
                    marginLeft: 8, padding: "2px 8px", borderRadius: 10,
                    background: SCORE_COLOR(selected.score),
                    color: "white", fontSize: 11, fontWeight: 700,
                  }}>
                    {(selected.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ fontSize: 12, color: "#444", marginBottom: 4 }}>{selected.deposit}</div>
                <div style={{ fontSize: 12 }}>Grade: <strong>{selected.grade_pct}% Cu</strong></div>
                <div style={{ fontSize: 12 }}>Viability: <strong style={{ color: SCORE_COLOR(selected.score) }}>{SCORE_LABEL(selected.score)}</strong></div>
                <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
                  {selected.lat.toFixed(4)}, {selected.lon.toFixed(4)}
                </div>
              </div>
            </Popup>
          )}
        </Map>
      </div>
    </div>
  );
}
