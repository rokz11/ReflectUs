import React from "react";

export default function BetweenUSLogo({ size = 40 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="BetweenUS logo"
    >
      <defs>
        <linearGradient id="betweenGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#7c3aed" />
          <stop offset="100%" stopColor="#a78bfa" />
        </linearGradient>
      </defs>

      <circle cx="42" cy="50" r="28" fill="rgba(124,58,237,0.6)" />
      <circle cx="58" cy="50" r="28" fill="rgba(167,139,250,0.6)" />

      <circle cx="50" cy="50" r="16" fill="url(#betweenGradient)" />
    </svg>
  );
}
