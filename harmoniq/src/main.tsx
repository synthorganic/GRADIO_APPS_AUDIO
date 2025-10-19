import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { LoopLibraryProvider } from "./state/LoopLibraryStore";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <LoopLibraryProvider>
      <App />
    </LoopLibraryProvider>
  </React.StrictMode>,
);
