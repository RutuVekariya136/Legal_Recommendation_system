import { afterEach, describe, expect, it, vi } from "vitest";

// A fake axios client whose .get/.post we can assert on.
const mockClient = { get: vi.fn(), post: vi.fn() };
vi.mock("axios", () => ({
  default: { create: vi.fn(() => mockClient) },
}));

const { api, apiError } = await import("../services/api.js");

afterEach(() => {
  mockClient.get.mockReset();
  mockClient.post.mockReset();
});

describe("api client", () => {
  it("recommendCase hits the right URL with params", async () => {
    mockClient.get.mockResolvedValue({ data: { results: [] } });
    await api.recommendCase("1955_S_34", { k: 5, yearMin: 1990 });
    expect(mockClient.get).toHaveBeenCalledWith(
      "/recommend/case/1955_S_34",
      expect.objectContaining({ params: expect.objectContaining({ k: 5, year_min: 1990 }) }),
    );
  });

  it("compare passes a and b", async () => {
    mockClient.get.mockResolvedValue({ data: {} });
    await api.compare("x", "y");
    expect(mockClient.get).toHaveBeenCalledWith("/compare", { params: { a: "x", b: "y" } });
  });

  it("egoGraph hits /graph/case/:id", async () => {
    mockClient.get.mockResolvedValue({ data: { nodes: [] } });
    await api.egoGraph("1990_A_1", { similar: 4 });
    expect(mockClient.get).toHaveBeenCalledWith(
      "/graph/case/1990_A_1",
      expect.objectContaining({ params: expect.objectContaining({ similar: 4 }) }),
    );
  });

  it("upload builds a FormData with the file and year", async () => {
    mockClient.post.mockResolvedValue({ data: { id: "abc" } });
    const file = new File(["hi"], "c.txt", { type: "text/plain" });
    const out = await api.upload(file, 2015);
    expect(out).toEqual({ id: "abc" });
    const [url, form] = mockClient.post.mock.calls[0];
    expect(url).toBe("/upload");
    expect(form).toBeInstanceOf(FormData);
    expect(form.get("file")).toBe(file);
    expect(form.get("year")).toBe("2015");
  });

  it("apiError prefers the FastAPI detail message", () => {
    expect(apiError({ response: { data: { detail: "nope" } } })).toBe("nope");
    expect(apiError({ message: "network" })).toBe("network");
    expect(apiError({})).toBe("Request failed");
  });
});
