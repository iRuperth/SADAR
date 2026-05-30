import { Route, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Presentation from "./pages/Presentation";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/presentation" element={<Presentation />} />
    </Routes>
  );
}
