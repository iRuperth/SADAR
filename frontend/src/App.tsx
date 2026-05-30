import { Route, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Presentation, { PresentationFinal } from "./pages/Presentation";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/presentation" element={<Presentation />} />
      <Route path="/presentation/final" element={<PresentationFinal />} />
    </Routes>
  );
}
