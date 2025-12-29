// src/pages/Results.js
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { API_BASE } from "../api";

export default function Results() {
  const { sessionId } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    const i = setInterval(async () => {
      const res = await fetch(`${API_BASE}/get_results?session_id=${sessionId}`);
      const json = await res.json();
      if (json.analysis) {
        setData(json);
        clearInterval(i);
      }
    }, 2500);

    return () => clearInterval(i);
  }, [sessionId]);

  return (
    <div className="fade-in" style={{ maxWidth: 680, margin: "0 auto" }}>
      
      {/* META TITLE */}
      <h2 style={{ marginBottom: 10 }}>Shared reflection</h2>

      <p className="reflection-frame">
        A calm synthesis of both perspectives.
      </p>

      <div className="card">
        {!data && (
          <p className="waiting-video-text">
            This may take a moment
          </p>
        )}

        {data && (
          <>
            {/* ACTUAL REFLECTION */}
            <p className="reflection-body">
              {data.analysis}
            </p>

            {/* CLOSING FRAME */}
            <p className="reflection-frame" style={{ marginTop: 28 }}>
              Individual answers remain private.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
