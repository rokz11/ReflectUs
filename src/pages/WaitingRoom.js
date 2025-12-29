import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { API_BASE } from "../api";
import "./WaitingRoom.css";

export default function WaitingRoom() {
  const { sessionId, role } = useParams();
  const navigate = useNavigate();

  const [readyA, setReadyA] = useState(false);
  const [readyB, setReadyB] = useState(false);
  const [hasMarkedReady, setHasMarkedReady] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/check_ready/${sessionId}`);
        const data = await res.json();

        setReadyA(Boolean(data.A_ready));
        setReadyB(Boolean(data.B_ready));

        if (data.both_ready) {
          clearInterval(interval);
          setTimeout(() => {
            navigate(`/results/${sessionId}`);
          }, 1500);
        }
      } catch (e) {
        console.error("check_ready failed");
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [sessionId, navigate]);

  async function markReady() {
    try {
      await fetch(`${API_BASE}/ready`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          role
        })
      });

      setHasMarkedReady(true);
    } catch (e) {
      alert("Failed to mark ready");
    }
  }

  const youReady = role === "A" ? readyA : readyB;
  const partnerReady = role === "A" ? readyB : readyA;

  return (
    <div className="waiting-container fade-in">
      <div className="waiting-card card">
        
        {/* Title in same style as onboarding */}
        <div className="onboarding-title">
          Waiting together
        </div>

        <p className="hint" style={{ marginBottom: 28 }}>
          This space will continue once both of you are ready.
        </p>

        <div className="waiting-status">
          <span>You</span>
          <span>{youReady ? "Ready" : "Not ready"}</span>
        </div>

        <div className="waiting-status">
          <span>Your partner</span>
          <span>{partnerReady ? "Ready" : "Not ready"}</span>
        </div>

        {!hasMarkedReady && (
          <button
            className="btn btn-primary"
            style={{ marginTop: 28 }}
            onClick={markReady}
          >
            I am ready
          </button>
        )}

        {hasMarkedReady && !partnerReady && (
          <p className="hint" style={{ marginTop: 20 }}>
            Please wait. Your partner will join shortly.
          </p>
        )}

        {youReady && partnerReady && (
          <p className="hint" style={{ marginTop: 20 }}>
            Thank you. The reflection is being prepared.
          </p>
        )}
      </div>
    </div>
  );
}
