# LSV1994 replication

The main workflow covers Q2-1 **Tables I, II, IV, V, VI and Figure 2**, with the original April 1968-1989 formations and returns through April 1990. Run from the project root:

~~~powershell
.\.venv\Scripts\python.exe main_lsv.py
~~~

Settings are at the bottom of main_lsv.py. Defaults use April 1968-1989
formations, returns ending April 1990, NYSE/AMEX common shares, and CEQ book
equity. No WRDS connection is opened. Table I uses the three original cached inputs and excludes the accounting outcome buffer from its signals. Table V additionally reads future accounting outcomes and April market equity from the separately selected extended legacy cache; this never extends the holding-return cutoff. Set run_replication=False to regenerate reports only.

The runner first calls replication/table1.py, then replication/workflow.py for the remaining selected tables. _formation.py constructs characteristics
and membership; _holding.py constructs annual returns and size benchmarks.
reporter/table1.py reads saved outputs and verifies file hashes, complete
cohort coverage, and agreement between summary and cohort returns before
reporting. Reports/index.html is self-contained and opens offline.

## Economic conventions

- At April t, use the last accounting record whose datadate falls in calendar
  year t-1. Resolve active CCM links at April formation, prioritizing P over C
  and LC over LU over LS. Keep one PERMNO per formation year. This matches the
  reference's timing convention; it is not a point-in-time accounting database.
- Market equity is abs(prc)*shrout/1000, in USD millions. Apply share codes
  10/11, the selected exchange codes, and positive market equity at formation.
  Do not require the same classifications during subsequent holding months.
- B/M=BE/ME, C/P=(IB+DP)/ME, E/P=IB/ME. Only C/P and E/P sorts require positive
  ratios. B/M retains all finite ratios as in the supplied notebook.
- Sales growth requires consecutive accounting years and positive current and
  previous sales. GS requires five annual growth ranks, normalized to within-year
  percentiles with average-rank ties, weighted 5,4,3,2,1 from recent to oldest.
- Deciles use SAS average ranks: floor(rank*10/(N+1))+1. Tied signals stay
  together. Empty deciles or incomplete cohort coverage cause an error rather
  than an apparently complete table.
- Hold May-April for years 1-5. Re-equal-weight surviving original constituents
  each April without re-sorting or admitting new stocks. Survival means a CRSP
  observation at the beginning April; later exchange changes do not remove returns.
- After the final observed CRSP row within a holding year, replace the remaining
  months with that stock's size-decile monthly benchmark. Missing returns before
  or at the last observed row are not filled. An annual stock return requires
  twelve valid monthly returns after this replacement. Missing annual returns
  are excluded from that cohort's stock average, as in the reference.
- Reassign size using the preceding December each holding year. Benchmarks use
  eligible April members, monthly equal-weighted replacement returns, and the
  average of members' annual buy-and-hold returns. They are not computed by
  compounding the monthly equal-weighted index.
- Average cohort returns separately at each horizon, giving 22/21/20/19/18
  cohorts in the original sample. AR averages R1-R5; CR5 compounds these five
  horizon means; SAAR averages the five size-adjusted horizon means. CR5 is
  distinct from the mean of complete-cohort five-year returns used in Table VI.

## Reference integrity and limitations

references/lsv1994_table1_reference_results.json keeps all 320 published values
and the historical notebook's separately labelled, three-decimal outputs. The
published values were checked against printed pp. 1548-1549. Comparisons are
attached only for the original legacy sample with the reference defaults.

The independent validator replays the supplied notebook's calculation cells
using the same local inputs, bypassing its WRDS connection and external cache
paths. It compares formation signals, membership, annual stock returns,
cohort returns, and all 320 summary cells at full precision. Its JSON record
contains input and notebook hashes and a check that raw inputs are unchanged.

The reference does not convert non-USD accounting values. The current baseline
has **403 CAD formation observations** whose accounting numerators are divided
by USD market equity. This is a currency inconsistency, not an exact measure of
those ratios. The baseline deliberately preserves it to reproduce the supplied
reference and flags it prominently in the report. Set accounting_currency="USD"
in main_lsv.py for a separately named sensitivity that excludes non-USD accounts;
this filter applies to historical GS accounting inputs too. No currency conversion
is inferred without an exchange-rate dataset.

Other limitations are current data vintage, calendar-year account selection,
the assumed disappearance proxy, and no separate legacy delisting-return
adjustment. Raw and size-adjusted returns can use different stock counts when
December size is unavailable. Counts and missing-return diagnostics are retained.
Matching the notebook does not establish exact agreement with the original paper.

## Outputs and validation

Each run lives under results/lsv1994/replication/table1/<run_tag> and retains:

- formation_panel.parquet, portfolio_constituents.parquet;
- annual_stock_returns.parquet, size_benchmark_returns.csv;
- portfolio_year_returns.csv, formation_counts.csv, horizon_coverage.csv;
- table1_summary.csv (320 rows), run_info.json (settings, provenance, diagnostics,
  limitations and output hashes);
- reports/table1.csv, reports/table1_comparison.csv, two PNG figures and the
  self-contained reports/index.html.

Run the focused tests:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.test_lsv1994_table1 tests.test_lsv1994_q21 tests.test_lsv1994_runs -v
~~~

For an independent full-precision notebook replay, use:

~~~powershell
.\.venv\Scripts\python.exe -m tests.validate_lsv1994_reference <raw_folder> <result_folder>
~~~

This writes reference_validation.json. Regenerating the report then shows the
validation result if its notebook and summary hashes match the saved run.

## All-table workflow and final submission

main_lsv.py selects replication_table=(1,2,4,5,6) by default. Table I is the explicit shared foundation; when running another table alone, a compatible saved Table I run must already exist. Intermediate pair memberships and annual stock/cohort returns are saved under data/processed/lsv1994/<run_tag>. Each table keeps its own results/lsv1994/replication/tableN/<run_tag> folder.

Submission generation is off by default in every runner. Use the saved-results command below to rebuild results/lsv1994/submission/submission.html with Q2-1–Q2-4. Setting build_submission=True in main_lsv.py instead creates the original-only preview submission_q21.html. Like the JT submission, this is one offline file with complete calculated tables, independent published-paper comparisons, five figures, discussion and sample filters. It has scrollable tables with sticky headings and expand controls, full-table printing, and embedded images. It does not require neighboring files, raw data, WRDS, or an external JavaScript library.

To assemble the submission from saved outputs without rerunning any portfolios or reading raw data, run:

~~~powershell
.\.venv\Scripts\python.exe submission_lsv1994.py
~~~

Edit core/submission/lsv1994_template.html for prose and section order. The existing shared submission.css and _html.py are reused without changing the JT submission. Frozen published benchmarks for all five tables are under references/lsv1994/paper_tables, independent of the historical notebook outputs.

### Added table conventions

- Table II: five independent double sorts; common eligible sample for each pair, SAS deciles collapsed 1-3/4-7/8-10, and all 360 return cells.
- Table IV: all nine displayed specifications and 32 coefficients; annual cross-sectional NumPy OLS, mean coefficients and time-series Fama-MacBeth standard errors; model-specific complete cases, missing ratios remain missing, no winsorization. Negative ratios use positive parts and negative indicators where the published model includes them. If a model-specific annual sample has no negative ratios, its all-zero indicator is omitted from that annual fit: other slopes are retained, while that dummy coefficient is NaN with status "no negative-ratio observations". Its time-series mean and inference exclude that year only for the unidentified term; n_years, n_total_years and n_unidentified_years are saved. At least two identified annual coefficients per term are required. Other rank deficiencies still raise an error.
- Table V: all 64 positions; current fundamentals average equal-weighted cohort means. Accounting year 0 is t-1. Each annual growth transition uses fixed ownership 1/April market equity at its base relative year, paired accounting observations and cross-cohort average levels before growth. Future accounts and April equity extend through 1993. Past return is mean complete individual 36-month buy-and-hold return. Price timing and missing-outcome treatment are explicit implementation conventions, not fully specified original-program choices.
- Table VI: compound cohort legs separately before subtracting; pool stocks across the two extreme deciles for equal stock weights. The main B/M definition follows the heading/text; the caption's single-decile alternative is retained as strategy B/M in the saved summary/cohorts. Complete 1/3/5-year cohorts number 22/20/18. HH uses uniform lag weights, lags 0/2/4 and n/(n-1) correction.
- Figure 2: exactly Table VI's annual C/P-GS series. D uses the May-April CRSP EW index return. R uses overlap of that holding window with the versioned NBER chronology in references/lsv1994/recessions.csv (peak and trough months inclusive); this can differ from the printed marker convention. The snapshot includes the 1990-1991, 2001, 2007-2009 and 2020 recessions and records its coverage/source in recessions.json. Runs beyond that verified coverage fail until the snapshot is updated. Per-table reports save and embed Figure 2 without needing the submission builder.

### Material numerical limitations

The inherited negative-book-equity inclusion produces a negative mean B/M for the calculated B/M glamour portfolio (about -0.048 versus the paper's 0.225). Ten of the 64 baseline Table V statistics are undefined because aggregate earnings/cash flow are nonpositive. They remain NaN with a status and appear as dashes; they are not represented as successful numerical replications. No extra filter or winsorization is chosen merely to force a match. The previously identified non-USD accounting issue also remains explicit.

The specified HH estimator does not reproduce every multi-year published t-statistic even on the paper's rounded cohort returns. Table VI saves published_inference_audit.csv for this check; paper t-statistics remain separate frozen values. One-year conventional tests match the printed t-statistics within their rounding. Multiyear t-statistic differences therefore cannot all be attributed to replication return differences.

Focused validation:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.test_lsv1994_table1 tests.test_lsv1994_q21 -v
~~~

The tests cover independent sorts, known annual OLS coefficients, aggregate-level growth, separate outcomes, required equity buffers, HH lag weights, cumulative-leg compounding, EW-index annotations, saved-only report generation, mixed samples and stale outputs. Table I's independent full-precision notebook replay remains available as described above.


## Q2-2 universes and Q2-3 calendar subsamples

The separate batch entry points use the shared broad raw extract, apply exchange and share-code eligibility at each April formation, and run **Tables I, II, IV, V, VI and Figure 2** for every case. They do not query WRDS or rebuild the submission.

~~~powershell
# Q2-2: NYSE only and NYSE/AMEX/Nasdaq, requested 1962-2025.
.\.venv\Scripts\python.exe main_lsv_universes.py --dry-run
.\.venv\Scripts\python.exe main_lsv_universes.py

# Q2-3: NYSE/AMEX, requested calendar windows 1963-1989 and 1990-2025.
.\.venv\Scripts\python.exe main_lsv_subperiods.py --dry-run
.\.venv\Scripts\python.exe main_lsv_subperiods.py

# Reject unavailable requested history/endpoints before starting any case.
.\.venv\Scripts\python.exe main_lsv_universes.py --strict-coverage

# Optional: run the subsample comparisons for both requested Q2-2 universes too.
.\.venv\Scripts\python.exe main_lsv_subperiods.py --universes nyse nyse_amex_nasdaq
~~~

Both commands accept --raw-data-dir, --end-year, --universes, --strict-coverage and --dry-run. Q2-2 also accepts --start-year. By default, the raw folder is data/raw/lsv1994/nyse_amex_nasdaq_19570101_20241231_comp19570101_20241231. Use --raw-data-dir to select a newly prepared cache, including CIZ data through 2025, without editing the runner. Every input requires its coverage JSON sidecar.

The currently cached stock/index series end in December 2024 and accounting history starts in 1957. Therefore, the available-data plans are:

| Case | Effective April formations | Complete May-April holding years | Last future accounting outcome year |
| --- | --- | --- | --- |
| Q2-2, either universe | 1963-2023 | May 1963-April 2024 | 2024 |
| Q2-3, early | 1963-1988 | May 1963-April 1989 | 1989 |
| Q2-3, recent | 1990-2023 | May 1990-April 2024 | 2024 |

These are explicit coverage adjustments, not completed 1962-2025 replications. Full 1962 GS formations need accounts from 1956; returns through 2025 need an updated stock/index cache. A complete 2025 cache permits the final complete holding year May 2024-April 2025. The annual paper definitions do not use a partial May-December holding year or annualize it.

The two Q2-3 periods are **calendar windows**. Preformation history is allowed for sales-growth signals, size, past growth and past returns. All subsequent holding returns and future accounting growth must fall within the window. This means the early run excludes April 1989 formations, which would require returns in 1990. Three- and five-year statistics use only complete horizons inside the window. The recent run starts with April 1990 formations and May 1990 returns; it does not import portfolios formed in the earlier period.

Table V keeps every formation cohort for current fundamentals and past-performance statistics. Future-growth statistics use only calendar-eligible cohorts for which the full five-year follow-up could be observed: formation t requires accounts and April market equity through t+4. The **same cohort set** is used for every future transition and both future-growth intervals. With outcomes ending in 1989/2024, the last eligible formation is 1985/2020. Firm-level accounting pairs remain pairwise complete, as in the original workflow. Missing calendar coverage causes an error; nonpositive aggregate levels or zero eligible future cohorts remain explicit NaNs with statuses and cohort counts. None of these outcome restrictions affects portfolio formation or holding returns.

Outputs retain the normal results/lsv1994/replication/tableN/<run_tag> structure. Batch tags start q22_ or q23_ and include the requested window, universe, CRSP source, effective formations and return endpoint, so they do not overwrite Q2-1. Each run_info.json stores its requested/effective dates, coverage adjustments and Table V policy; the per-run HTML reports display the coverage plan. The shared runner is core/lsv1994/runs.py. Submission authoring is an explicit separate action.


## Assembled Q2 submission

submission_lsv1994.py explicitly selects the original Q2-1 run and the four completed Q2-2/Q2-3 cases. The builder validates each saved-output hash, table key set, common Table I foundation, raw-data vintage, accounting convention, complete Table VI cohorts and Figure 2 alignment. Extended cases must match the requested universes/windows; Table V outcome endpoints and cohort counts are checked against those windows. It never queries WRDS or reconstructs portfolios.

The single offline submission includes all five tables for all five cases, original-paper comparisons for Q2-1, universe and subsample spread comparisons, 15 embedded figures, the Q2-3 interpretation and Q2-4 filters. The two added sections use the JT-style case selector, with all hidden cases included when printing. Table captions/controls have unique identifiers. A coverage table at the beginning clearly distinguishes the requested windows from actual formation/return dates and future-growth cohort counts. It explicitly identifies missing 1956 warm-up accounts and 2025 stock/index data.

The Q2-3 discussion distinguishes Table I AR/SAAR spreads from Table VI annual tail spreads. Table I AR decile spreads decline for all four signals in the recent run; Table VI annual mean changes are mixed across C/P tails, C/P-GS and B/M tails. The reported t-statistics describe each mean rather than testing a difference between subsamples. Undefined accounting growth and unestimated annual negative-ratio dummies remain visible in the case notes.

The saved-results builder writes submission_manifest.json alongside the HTML, recording its hash, template/builder hashes, case settings, coverage plans and input-output hashes. The prior Q2-1 HTML is preserved as submission_q21.html. The JT submission and shared submission.css/_html.py remain unchanged.
