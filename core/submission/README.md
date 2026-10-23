# JT1993 submission

From the project root, run:

```powershell
.venv/Scripts/python.exe -B submission_jt1993.py
```

This produces `results/jt1993/submission/submission.html`, a single offline file
covering the available HW1 Q1-1 through Q1-5 materials: original-paper comparisons,
exchange comparisons, subperiod comparisons, complete Tables I, II, and VII,
nine figures, discussion, and the baseline sample-selection and data-availability
rules with a link to CRSP's share-code definitions, and the saved FF3 regression.
Q1-5 includes the regression equation, four coefficients, HAC standard errors,
t-statistics, p-values, 95% intervals, N, R-squared, and a data-driven conclusion.
Each long table has a fixed-height
scrolling panel, sticky column headings, and an expand/collapse control. Tables
print in full. The submission does not include data exports or a supporting-data
section.
No external fonts, JavaScript libraries, images, or neighboring files are needed
to view it. There is no separate run/build-info output.

Select the saved table/sample directories, `ff3_result_dir`, and the output path in
`submission_jt1993.py`. The generator reads each summary CSV and uses its existing
`run_info.json` internally to reject mixed samples and label Table VII's endpoint
convention. Q1-5 must match the selected original Table I universe, dates, and
source-return checksum. Its table is read from the full-precision saved FF3
coefficients; alpha is converted to percent per month for display. The original
regression's five-lag Bartlett/Newey-West and asymptotic-normal inference settings
are stated explicitly. It does not read raw stock data, connect to WRDS, rerun replication,
or modify any analysis outputs. Paper comparisons require the original NYSE/AMEX
January 1965-December 1989 sample. Exchange and period comparisons use their
actual saved dates and do not use paper/notebook benchmark columns.

The selected exchange runs currently cover 1965-1989 for NYSE, NYSE/AMEX, and
NYSE/AMEX/Nasdaq. The selected NYSE/AMEX subperiods are 1965-1989 and 1990-2024.
These are narrower than the homework's 1962-present request; raw data through
2024 do not imply that every analysis has been run for that entire span. The
submission labels this coverage explicitly. No missing analyses are fabricated
or reconstructed by averaging subsample summaries.

Each extension shows side-by-side numerical comparisons and graphs. A selector
under the complete-results disclosure switches between full individual tables.
Exchange groups must have matching windows and distinct universes. Subperiods
must share a universe and be chronological and adjacent. Data-source and method
conventions must agree within each comparison.

Published numbers are maintained in `references/jt1993/paper_tables/`; consult
that directory's README for pages, units, keys, and transcription conventions.
They are independent of the WRDS notebook reference. Any benchmark columns
already present in a saved summary are ignored.

Edit `jt1993_template.html` for section order and written discussion, and
`submission.css` for appearance. Numerical values in the discussion are inserted by
`jt1993.py`, so they update when selected results change. Rebuild after edits;
direct changes to the generated HTML will be overwritten. Add later homework
analyses here as their saved results become available. Omit `ff3_result_dir` when
calling the generator to build without Q1-5. Tables II and VII retain
their draft construction notes.

Run the focused offline checks with:

```powershell
.venv/Scripts/python.exe -B -m unittest tests.test_jt1993_submission -v
```
