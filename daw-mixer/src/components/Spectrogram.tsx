import { useEffect, useRef } from "react";
import { audioEngine } from "../lib/audioEngine";
import { theme } from "../theme";

interface SpectrogramProps {
  height?: number;
  width?: number;
}

export function Spectrogram({ height = 120, width = 480 }: SpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;
    const analyser = audioEngine.getMasterAnalyser();
    const bins = new Uint8Array(analyser.frequencyBinCount);

    const draw = () => {
      analyser.getByteFrequencyData(bins);
      // Scroll image left by 1px
      const image = ctx.getImageData(1, 0, canvas.width - 1, canvas.height);
      ctx.putImageData(image, 0, 0);
      // Draw new column at right
      for (let y = 0; y < canvas.height; y += 1) {
        const bin = Math.floor((y / canvas.height) * bins.length);
        const v = bins[bin] ?? 0; // 0..255
        const hue = 200 + (v / 255) * 140; // bluish to greenish
        const light = 12 + (v / 255) * 50;
        ctx.fillStyle = `hsl(${hue} 70% ${light}%)`;
        ctx.fillRect(canvas.width - 1, canvas.height - y - 1, 1, 1);
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    // Clear
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    rafRef.current = requestAnimationFrame(draw);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  return (
    <div style={{
      border: `1px solid ${theme.border}`,
      borderRadius: "10px",
      background: theme.surfaceOverlay,
      boxShadow: theme.cardGlow,
      padding: "8px",
      width: `${width + 16}px`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
        <strong style={{ fontSize: "0.75rem", letterSpacing: "0.05em" }}>Spectrogram</strong>
        <span style={{ fontSize: "0.65rem", color: theme.textMuted }}>Master</span>
      </div>
      <canvas ref={canvasRef} width={width} height={height} style={{ display: "block", width: `${width}px`, height: `${height}px` }} />
    </div>
  );
}

