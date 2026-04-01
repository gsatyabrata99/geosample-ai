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
          "raster-opacity": 0.45,
          "raster-fade-duration": 300,
          "raster-brightness-min": 0.05,
          "raster-contrast": 0.3,
          "raster-saturation": 0.5,
        }}
      />
    </Source>
  );
}
