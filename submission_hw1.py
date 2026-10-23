"""Combine the existing JT and LSV HTML submissions without recalculating results."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def combine_submissions():
    sources = [
        ROOT / "results/jt1993/submission/submission.html",
        ROOT / "results/lsv1994/submission/submission.html",
    ]
    parts, styles = [], []
    for paper, source in zip(("jt1993", "lsv1994"), sources):
        html = source.read_text(encoding="utf-8")
        styles.extend(re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.S | re.I))
        body = re.search(r"<body\b[^>]*>(.*?)</body>", html, re.S | re.I).group(1)
        if paper == "jt1993":
            script = re.search(r"<script\b[^>]*>(.*?)</script>", body, re.S | re.I).group(1)
        body = re.sub(r"<script\b[^>]*>.*?</script>", "", body, flags=re.S | re.I)
        if paper == "lsv1994":
            # The reports reuse section IDs. Update LSV's anchors, labels, and
            # case selectors together so the shared controls remain independent.
            ids = {value: f"lsv-{value}" for value in re.findall(r'\bid="([^"]+)"', body)}
            attributes = re.compile(
                r'(?P<start>\s(?:id|for|aria-controls|aria-labelledby|data-group|value)=")'
                r'(?P<value>[^"]*)(?P<end>")'
            )
            body = attributes.sub(
                lambda match: match["start"] + " ".join(
                    ids.get(value, value) for value in match["value"].split()
                ) + match["end"], body,
            )
            body = re.sub(r'href="#([^"]+)"', lambda match: f'href="#{ids.get(match[1], match[1])}"', body)
        # Retain each report's existing content width without nesting main elements.
        body = body.replace("<main>", '<div class="paper-content">').replace("</main>", "</div>")
        parts.append(f'<article class="submission-part" id="{paper}">\n{body.strip()}\n</article>')

    style = "\n".join(dict.fromkeys(value.strip() for value in styles))
    presentation = """
.submission-nav { max-width: none; min-height: 54px; padding: 12px max(22px, calc((100vw - 1076px) / 2));
  gap: 12px 24px; background: #203337; border: 0; }
.submission-nav .submission-title { margin-right: auto; color: #e3f1ec; font-size: 13px; font-weight: 600; }
.submission-nav a { color: #e3f1ec; }
.submission-part > nav { top: 54px; }
.submission-part + .submission-part { margin-top: 48px; }
.paper-content { max-width: 1160px; margin: auto; padding: 0 20px; }
html { scroll-padding-top: 150px; }
@media (max-width: 760px) {
  .submission-nav { gap: 10px 16px; padding-inline: 16px; }
  .submission-nav a { font-size: 13px; }
  .paper-content { padding: 0 10px; }
  html { scroll-padding-top: 200px; }
}
@media print {
  .submission-nav { display: none; }
  .paper-content { padding: 0; }
  .submission-part + .submission-part { break-before: page; margin-top: 0; }
}
"""
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phil Xu · Homework 1 · Momentum and Value Replications</title>
<style>{style}\n{presentation}</style></head>
<body>
<nav class="submission-nav" aria-label="Homework questions">
  <span class="submission-title">Homework 1 · Phil Xu</span>
  <a href="#jt1993">Q1 · JT1993</a><a href="#lsv1994">Q2 · LSV1994</a>
</nav>
{chr(10).join(parts)}
<script>{script}</script>
</body></html>
"""
    output = ROOT / "results/submission/hw1_submission.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(f"Combined submission saved: {combine_submissions()}")
