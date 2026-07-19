import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev workflow: `tokenlens server --port 8321` in one terminal, `npm run dev`
// in another. The proxy forwards API + WebSocket calls so the app can use
// same-origin relative URLs in both dev and production builds.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8321", changeOrigin: true },
      "/ws": { target: "ws://localhost:8321", ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
