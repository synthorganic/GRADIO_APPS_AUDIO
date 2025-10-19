import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@src": fileURLToPath(new URL("./src", import.meta.url)),
      "@daw-shared": fileURLToPath(new URL("../daw-mixer/src/shared", import.meta.url)),
      "@daw-theme": fileURLToPath(new URL("../daw-mixer/src/theme.ts", import.meta.url))
    }
  },
  server: {
    host: "0.0.0.0",
    port: 4174,
    fs: {
      allow: [".."]
    }
  }
});
