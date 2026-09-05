import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import CasePage from "./pages/CasePage.jsx";
import ComparePage from "./pages/ComparePage.jsx";
import HomePage from "./pages/HomePage.jsx";
import RecommendPage from "./pages/RecommendPage.jsx";
import SearchPage from "./pages/SearchPage.jsx";
import UploadPage from "./pages/UploadPage.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/case/:docId" element={<CasePage />} />
        <Route path="/recommend/case/:docId" element={<RecommendPage kind="case" />} />
        <Route path="/recommend/upload/:uploadId" element={<RecommendPage kind="upload" />} />
        <Route path="/compare" element={<ComparePage kind="pair" />} />
        <Route path="/compare/upload/:uploadId" element={<ComparePage kind="upload" />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<p>Not found.</p>} />
      </Routes>
    </Layout>
  );
}
