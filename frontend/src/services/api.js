import axios from "axios";

// Local dev: Vite proxies /api to the backend (see vite.config.js).
// Production (e.g. Vercel): uses VITE_API_BASE_URL pointing to the Render backend.
const baseURL = import.meta.env.VITE_API_BASE_URL || "/api";
const client = axios.create({ baseURL, timeout: 60000 });

function unwrap(p) {
  return p.then((r) => r.data);
}

export const api = {
  ready: () => unwrap(client.get("/health/ready")),
  stats: () => unwrap(client.get("/stats")),
  evalSummary: () => unwrap(client.get("/evaluate/summary")),

  search: (q, { mode = "semantic", yearMin, yearMax, split, limit = 20, offset = 0, withFacets = true } = {}) =>
    unwrap(
      client.get("/search", {
        params: {
          q, mode, year_min: yearMin, year_max: yearMax, split,
          limit, offset, with_facets: withFacets,
        },
      }),
    ),
  facets: () => unwrap(client.get("/search/facets")),

  listCases: ({ q, yearMin, yearMax, limit = 50, offset = 0 } = {}) =>
    unwrap(client.get("/cases", { params: { q, year_min: yearMin, year_max: yearMax, limit, offset } })),
  getCase: (docId) => unwrap(client.get(`/cases/${encodeURIComponent(docId)}`)),
  getCaseText: (docId) => unwrap(client.get(`/cases/${encodeURIComponent(docId)}/text`)),
  getCaseTreatment: (docId) => unwrap(client.get(`/cases/${encodeURIComponent(docId)}/treatment`)),

  recommendCase: (docId, { k = 10, yearMin, yearMax, minScore } = {}) =>
    unwrap(
      client.get(`/recommend/case/${encodeURIComponent(docId)}`, {
        params: { k, year_min: yearMin, year_max: yearMax, min_score: minScore },
      }),
    ),
  recommendUpload: (uploadId, { k = 10, yearMin, yearMax, minScore, category, categoryQuery, details } = {}) =>
    unwrap(
      client.post(`/recommend/upload/${encodeURIComponent(uploadId)}`, details && Object.keys(details).length ? { details } : null, {
        params: { k, year_min: yearMin, year_max: yearMax, min_score: minScore,
                  category, category_query: categoryQuery },
      }),
    ),

  uploadParameters: (uploadId, category) =>
    unwrap(client.get(`/upload/${encodeURIComponent(uploadId)}/parameters`, { params: { category } })),
  parameterImpact: () => unwrap(client.get("/evaluate/parameters")),

  egoGraph: (docId, { similar = 6, statutes = 8 } = {}) =>
    unwrap(client.get(`/graph/case/${encodeURIComponent(docId)}`, { params: { similar, statutes } })),

  compare: (a, b) => unwrap(client.get("/compare", { params: { a, b } })),
  compareUpload: (uploadId, target) =>
    unwrap(client.get(`/compare/upload/${encodeURIComponent(uploadId)}`, { params: { target } })),

  upload: (file, year, category, categoryQuery) => {
    const form = new FormData();
    form.append("file", file);
    if (year) form.append("year", year);
    if (category) form.append("category", category);
    if (categoryQuery) form.append("category_query", categoryQuery);
    return unwrap(client.post("/upload", form, { headers: { "Content-Type": "multipart/form-data" } }));
  },
  getUpload: (id) => unwrap(client.get(`/upload/${encodeURIComponent(id)}`)),
  categories: () => unwrap(client.get("/categories")),
};

export function apiError(err) {
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d?.msg || String(d)).join("; ") || "Request failed";
  }
  if (detail && typeof detail === "object") return detail.msg || JSON.stringify(detail);
  return detail || err?.message || "Request failed";
}
