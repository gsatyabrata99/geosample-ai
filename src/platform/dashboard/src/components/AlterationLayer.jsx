import { useEffect, useState } from "react";
import { Source, Layer } from "react-map-gl";

export default function AlterationLayer({ visible }) {
  const [bounds, setBounds] = useState(null);

  useEffect(() => {
    fetch("/alteration_bounds.json")
      .then(r => r.json())
      .then(setBounds)
      .catch(() => {});
  }, []);

  if (!visible || !bounds) return null;

  return (
    <Source
      id="alteration"
      type="image"
      url="/alteration_score.png"
      coordinates={bounds.coordinates}
    >
      <Layer
        id="alteration-layer"
        type="raster"
        paint={{
          "raster-opacity": 0.6,
          "raster-fade-duration": 300,
        }}
      />
    </Source>
  );
}
