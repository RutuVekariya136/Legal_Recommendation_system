import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import { Loading } from "./components/ui.jsx";

// Route-level code splitting keeps the initial bundle small (Recharts / graph
// view load only on the pages that use them).
const HomePage = lazy(() => import("./pages/HomePage.jsx"));
const SearchPage = lazy(() => import("./pages/SearchPage.jsx"));
const CasePage = lazy(() => import("./pages/CasePage.jsx"));
const RecommendPage = lazy(() => import("./pages/RecommendPage.jsx"));
const ComparePage = lazy(() => import("./pages/ComparePage.jsx"));
const GraphPage = lazy(() => import("./pages/GraphPage.jsx"));
const UploadPage = lazy(() => import("./pages/UploadPage.jsx"));
const AboutPage = lazy(() => import("./pages/AboutPage.jsx"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage.jsx"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/case/:docId" element={<CasePage />} />
          <Route path="/case/:docId/graph" element={<GraphPage />} />
          <Route path="/recommend/case/:docId" element={<RecommendPage kind="case" />} />
          <Route path="/recommend/upload/:uploadId" element={<RecommendPage kind="upload" />} />
          <Route path="/compare" element={<ComparePage kind="pair" />} />
          <Route path="/compare/upload/:uploadId" element={<ComparePage kind="upload" />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<p>Not found.</p>} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
