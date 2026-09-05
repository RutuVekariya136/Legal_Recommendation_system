import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScoreBar, Badge } from "../components/ui.jsx";
import HighlightedText from "../components/HighlightedText.jsx";

describe("ScoreBar", () => {
  it("renders the label and value, clamps the bar width", () => {
    render(<ScoreBar label="hybrid" value={0.42} />);
    expect(screen.getByText("hybrid")).toBeInTheDocument();
    expect(screen.getByText("0.420")).toBeInTheDocument();
  });

  it("handles missing value", () => {
    render(<ScoreBar label="x" />);
    expect(screen.getByText("0.000")).toBeInTheDocument();
  });
});

describe("Badge", () => {
  it("renders children", () => {
    render(<Badge tone="green">s.302</Badge>);
    expect(screen.getByText("s.302")).toBeInTheDocument();
  });
});

describe("HighlightedText", () => {
  it("highlights shared statute refs and party keywords", () => {
    render(
      <HighlightedText
        text="The court applied s. 302 and cited Bolam on the standard of care."
        sections={["302"]}
        partyKeywords={["bolam"]}
      />,
    );
    const marks = document.querySelectorAll("mark");
    const texts = [...marks].map((m) => m.textContent);
    expect(texts).toContain("s. 302");
    expect(texts.map((t) => t.toLowerCase())).toContain("bolam");
  });

  it("renders plain text when nothing to highlight", () => {
    render(<HighlightedText text="nothing special here" />);
    expect(screen.getByText("nothing special here")).toBeInTheDocument();
    expect(document.querySelectorAll("mark")).toHaveLength(0);
  });
});
