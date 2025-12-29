import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function FreeExpression() {
  const { sessionId, role } = useParams();
  const navigate = useNavigate();
  const [text, setText] = useState("");

  function continueFlow() {
    if (!text.trim()) {
      if (
        !window.confirm(
          "You haven't written anything yet. Do you want to continue anyway?"
        )
      ) {
        return;
      }
    }

    // kasneje bomo to shranili v backend kot "free_expression"
    navigate(`/flow/${sessionId}/${role}`);
  }

  return (
    <div className="fade-in app-container">
      <div className="card">
        <div className="header-row">
          <div className="avatar" />
          <div>
            <h2>Before we begin</h2>
            <p className="hint">
              This space is only for you.
            </p>
          </div>
        </div>

        <p>
          Take a moment to write anything that feels important right now.
          <br />
          Why are you here? What hurts? What feels unresolved?
        </p>

        <p className="hint">
          You don’t need to be polite, structured, or careful.
          <br />
          Your partner will never see this.
        </p>

        <textarea
          className="input"
          placeholder="Write freely. There is no limit."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          className="btn btn-primary"
          style={{ marginTop: 16 }}
          onClick={continueFlow}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
