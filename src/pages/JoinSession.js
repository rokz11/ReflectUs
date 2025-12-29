import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_BASE } from "../api";

export default function JoinSession() {
  const [sessionId, setSessionId] = useState("");
  const [name, setName] = useState("");
  const navigate = useNavigate();

  async function join() {
    if (!sessionId || !name) return alert("Enter code and name.");

    const res = await fetch(`${API_BASE}/join_session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, name })
    });

    const data = await res.json();
    if (data.role === "B") {
      navigate(`/onboarding/${sessionId}/B`);
    } else {
      alert("Session not found");
    }
  }

  return (
    <div className="fade-in card form">
      <h2>Join session</h2>

      <input
        className="input"
        placeholder="Session code"
        value={sessionId}
        onChange={e => setSessionId(e.target.value)}
      />

      <input
        className="input"
        placeholder="Your display name"
        value={name}
        onChange={e => setName(e.target.value)}
      />

      <button className="btn btn-primary" onClick={join}>
        Join
      </button>
    </div>
  );
}
