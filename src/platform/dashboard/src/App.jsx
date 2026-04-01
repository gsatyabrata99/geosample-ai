import { useState, useEffect } from "react";
import axios from "axios";
import { MapPin, Activity, FileText, Cpu, TrendingUp, AlertCircle } from "lucide-react";
import SiteMap from "./components/SiteMap";
import PredictionPanel from "./components/PredictionPanel";
import ModelStats from "./components/ModelStats";
import ReportUpload from "./components/ReportUpload";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [health, setHealth] = useState(null);
  const [sites, setSites] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/health`),
      axios.get(`${API}/sites?min_score=0.0&limit=50`),
      axios.get(`${API}/model/info`),
    ]).then(([h, s, m]) => {
      setHealth(h.data);
      setSites(s.data.sites);
      setModelInfo(m.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const tabs = [
    { id: "map", label: "Site Map", icon: MapPin },
    { id: "predict", label: "Analyze Text", icon: FileText },
    { id: "upload", label: "Upload Report", icon: FileText },
    { id: "model", label: "Model Info", icon: Cpu },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <Activity size={20} />
            <span>GeoSample AI</span>
          </div>
          <span className="tagline">Copper Exploration Intelligence</span>
        </div>
        <div className="header-right">
          {health && (
            <div className="health-badges">
              {Object.entries(health.models).map(([k, v]) => (
                <span key={k} className={`badge ${v ? "badge-ok" : "badge-off"}`}>
                  {k.toUpperCase()}
                </span>
              ))}
            </div>
          )}
          <span className="region-tag">DRC / Zambia Copperbelt</span>
        </div>
      </header>

      <nav className="nav">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-btn ${activeTab === id ? "active" : ""}`}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </nav>

      <main className="main">
        {loading ? (
          <div className="loading">
            <Activity size={32} className="spin" />
            <p>Loading models...</p>
          </div>
        ) : (
          <>
            {activeTab === "map" && <SiteMap sites={sites} />}
            {activeTab === "predict" && <PredictionPanel api={API} />}
            {activeTab === "upload" && <ReportUpload api={API} />}
            {activeTab === "model" && <ModelStats modelInfo={modelInfo} health={health} />}
          </>
        )}
      </main>
    </div>
  );
}
