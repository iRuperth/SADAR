import { Route, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Presentation, { PresentationFinal } from "./pages/Presentation";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Presentation />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/presentation/final" element={<PresentationFinal />} />
    </Routes>
  );
}
