// src/pages/PartnerFlow.js
import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { API_BASE } from "../api";

const PROMPTS = [
  {
    id: 1,
    text:
      "What feels most present in your relationship right now?\n\nWhat has been taking up the most emotional space for you lately?"
  },
  {
    id: 2,
    text:
      "Why does this matter to you?\n\nWhat makes this feel important or hard to ignore?"
  },
  {
    id: 3,
    text:
      "Is there something unresolved, heavy, or quietly painful — even if it’s hard to put into words?"
  },
  {
    id: 4,
    text:
      "What would you want your partner to truly understand about you, if you could be heard without interruption?"
  },
  {
    id: 5,
    optional: true,
    text:
      "Before finishing, is there anything you would want to add?\n\nThis is optional."
  }
];

export default function PartnerFlow() {
  const { sessionId, role } = useParams();
  const navigate = useNavigate();

  const [step, setStep] = useState(0);
  const [answer, setAnswer] = useState("");

  const current = PROMPTS[step];

  async function next(skip = false) {
    if (!skip && !answer.trim()) return;

    await fetch(`${API_BASE}/save_answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        role,
        answer: answer.trim() || "[Skipped]"
      })
    });

    setAnswer("");

    if (step + 1 < PROMPTS.length) {
      setStep(step + 1);
    } else {
      navigate(`/wait/${sessionId}/${role}`);
    }
  }

  return (
    <div className="fade-in breathing-container">
      <div className="flow-wrapper">
        <h2 className="onboarding-title breathing-text">
          Take your time.
        </h2>

        <div className="card breathing-card">
          <p style={{ whiteSpace: "pre-line", fontSize: 16 }}>
            {current.text}
          </p>

          <textarea
            className="input"
            rows={8}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            style={{ marginTop: 22 }}
          />

          <div style={{ marginTop: 28, textAlign: "right" }}>
            {current.optional && (
              <button
                className="btn btn-ghost"
                onClick={() => next(true)}
                style={{ marginRight: 10 }}
              >
                Skip
              </button>
            )}

            <button
              className="btn btn-primary"
              onClick={() => next(false)}
            >
              {step + 1 === PROMPTS.length ? "Finish" : "Continue"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
