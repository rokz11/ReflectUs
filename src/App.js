// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Header from "./components/Header";

import CreateSession from "./pages/CreateSession";
import JoinSession from "./pages/JoinSession";
import Onboarding from "./pages/Onboarding";
import PartnerFlow from "./pages/PartnerFlow";
import WaitingRoom from "./pages/WaitingRoom";
import Results from "./pages/Results";
import About from "./pages/About";

import "./App.css";

export default function App() {
  return (
    <Router>
      <div className="app-root">
        <Header />

        <main className="app-main">
          <Routes>
            <Route path="/" element={<CreateSession />} />
            <Route path="/join" element={<JoinSession />} />
            <Route path="/about" element={<About />} />

            {/* ✅ PRAVILEN FLOW */}
            <Route path="/onboarding/:sessionId/:role" element={<Onboarding />} />
            <Route path="/flow/:sessionId/:role" element={<PartnerFlow />} />

            <Route path="/wait/:sessionId/:role" element={<WaitingRoom />} />
            <Route path="/results/:sessionId" element={<Results />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
