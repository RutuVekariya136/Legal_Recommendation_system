import axios from "axios";

const client = axios.create({ baseURL: "/api", timeout: 60000 });

function unwrap(p) {
  return p.then((r) => r.data);
}

export const api = {
  ready: () => unwrap(client.get("/health/ready")),
  stats: () => unwrap(client.get("/stats")),

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

  recommendCase: (docId, { k = 10, yearMin, yearMax, minScore } = {}) =>
    unwrap(
      client.get(`/recommend/case/${encodeURIComponent(docId)}`, {
        params: { k, year_min: yearMin, year_max: yearMax, min_score: minScore },
      }),
    ),
  recommendUpload: (uploadId, { k = 10, yearMin, yearMax, minScore } = {}) =>
    unwrap(
      client.post(`/recommend/upload/${encodeURIComponent(uploadId)}`, null, {
        params: { k, year_min: yearMin, year_max: yearMax, min_score: minScore },
      }),
    ),

  compare: (a, b) => unwrap(client.get("/compare", { params: { a, b } })),
  compareUpload: (uploadId, target) =>
    unwrap(client.get(`/compare/upload/${encodeURIComponent(uploadId)}`, { params: { target } })),

  upload: (file, year) => {
    const form = new FormData();
    form.append("file", file);
    if (year) form.append("year", year);
    return unwrap(client.post("/upload", form, { headers: { "Content-Type": "multipart/form-data" } }));
  },
  getUpload: (id) => unwrap(client.get(`/upload/${encodeURIComponent(id)}`)),
};

export function apiError(err) {
  return err?.response?.data?.detail || err?.message || "Request failed";
}
