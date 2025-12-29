import { useParams, useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Questions() {
  const { sessionId, partner } = useParams();
  const navigate = useNavigate();
  const [answer, setAnswer] = useState("");

  return (
    <div className="page">
      <h2>One important question</h2>
      <p className="muted">
        Answer honestly. Your partner will not see this.
      </p>

      <textarea
        placeholder="What feels hardest in your relationship right now?"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
      />

      <button
        className="btn-primary"
        onClick={() => navigate(`/wait/${sessionId}`)}
      >
        Finish
      </button>
    </div>
  );
}
