// src/pages/Onboarding.js
import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function Onboarding() {
  const { sessionId, role } = useParams();
  const navigate = useNavigate();
  const [step, setStep] = useState(1);

  function next() {
    setStep(s => s + 1);
  }

  function start() {
    navigate(`/flow/${sessionId}/${role}`);
  }

  return (
    <div className="fade-in">
      {step === 1 && (
        <div className="card">
          <h2 className="onboarding-title">Confidential space</h2>

          <p className="hint">
            Your answers are private and will never be shown to your partner.
          </p>

          <p>
            What you share here is used only to understand emotional patterns,
            not individual statements.
          </p>

          <button className="btn btn-primary" onClick={next}>
            I understand
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <h2 className="onboarding-title">Honesty matters</h2>

          <p className="hint">
            This process works only with honesty.
          </p>

          <p>
            You do not need to soften your words or protect anyone here.
            Even difficult thoughts are welcome.
          </p>

          <button className="btn btn-primary" onClick={next}>
            Continue
          </button>
        </div>
      )}

      {step === 3 && (
        <div className="card">
          <h2 className="onboarding-title">Your reflection</h2>

          <p className="hint">
            This space is yours.
          </p>

          <p>
            Write in your own words and take the space you need.
            Longer, more thoughtful responses help create a clearer reflection.
          </p>

          <p className="hint" style={{ marginTop: 12 }}>
            You may write in your native language if that helps you express yourself more clearly.
          </p>

          <button className="btn btn-primary" onClick={start}>
            Start
          </button>
        </div>
      )}
    </div>
  );
}
