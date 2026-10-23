# HW1 replication-results audit

Reviewed on 23 October 2026 against `docs/requirements.md`, the two current submission HTML files, their selected saved outputs, the implementation, the tests, and the supplied papers.

**Assessment: the project is a substantial, reproducible replication, but the current results do not satisfy all HW1 requirements.** The clearest omissions are the requested extended sample windows. LSV Table V also has genuinely unavailable requested statistics, while several original-paper numerical differences remain unresolved. Disclosure of these limitations is good research practice; disclosure alone does not complete the missing work.

## Requirement-by-requirement assessment

| Requirement | Assessment | Evidence and remaining work |
| --- | --- | --- |
| Q1-1: JT Tables I, II, VII, original universe/period, tables and graphs | Substantially covered; Tables II/VII remain provisional | All tables and figures exist for NYSE/AMEX, 1965-1989. Table I matches the supplied notebook to numerical precision. Tables II/VII reproduce broad patterns but use documented choices that have not been independently established as the original implementation. |
| Q1-2: repeat all tests for NYSE and NYSE/AMEX/Nasdaq, 1962-present | Not satisfied | Both exchange extensions report only 1965-1989. No saved full-window 1962-present Table I/II/VII runs exist for either required universe. A cache covering 1960-2024 is not a completed analysis of that span. |
| Q1-3: partition the entire 1962-present sample; assess recent momentum | Partly satisfied | All three tables and an interpretation exist for NYSE/AMEX, 1965-1989 versus 1990-2024. The beginning and ending dates do not cover the requested entire sample. |
| Q1-4: list stock filters | Satisfied in substance | The submission specifies exchanges, ordinary-common-share codes, historical classification timing, valid ranking histories, return validity, and table-specific availability rules. |
| Q1-5: FF3 regression for a spread strategy; evaluate alpha | Satisfied | Saved Panel A (6,6), January 1965-December 1989, 300 observations. Positive and significant alpha; independently checked OLS and HAC covariance. HW1 does not explicitly require repeating this regression for every sample. |
| Q2-1: LSV Tables I, II, IV, V, VI and Figure 2, original sample | Partly satisfied | Every requested component is present. Table I matches the supplied notebook; Table V has 10 undefined cells out of 64. Table IV loses some published significance findings, and Table VI multi-year inference differs materially. |
| Q2-2: repeat all tests for NYSE and NYSE/AMEX/Nasdaq, 1962-present | Partly satisfied | All five tables and Figure 2 exist for both universes, but formations are 1963-2023 and returns end in April 2024. Neither the 1962 beginning nor the requested latest endpoint is achieved. |
| Q2-3: 1963-1989 versus 1990-2025; assess weaker value | Partly satisfied | All tables, figures, and a nuanced comparison exist, but recent returns end in April 2024. The complete-annual-window convention also excludes the April 1989 formation and leaves a gap at the subsample boundary. |
| Q2-4: list stock filters | Satisfied in substance | The submission documents formation eligibility, accounting/linking, positive C/P and E/P sorting, GS history, missing returns, currencies, and table-specific availability. Adding the accounting database population filters would make the list more exhaustive. |

## 1. Sample coverage is the principal compliance gap

The selections in `submission_jt1993.py` are the actual analyses used in the current Q1 submission:

- Original: NYSE/AMEX, January 1965-December 1989.
- Exchange comparisons: NYSE, NYSE/AMEX, and NYSE/AMEX/Nasdaq, each January 1965-December 1989.
- Subperiods: NYSE/AMEX, January 1965-December 1989 and January 1990-December 2024.

Thus Q1-2 is missing the requested full-period analyses, even for dates already present in the cached data. Q1-3 supplies a valid comparison for its stated dates, but does not partition the entire requested window. There are five selected saved runs per table; none is a required full-period NYSE or all-exchange run.

The LSV submission manifest explicitly records the following effective coverage:

| Case | April formation years | Complete May-April holding-return coverage |
| --- | --- | --- |
| Original Q2-1 | 1968-1989 | May 1968-April 1990 |
| Q2-2, either universe | 1963-2023 | May 1963-April 2024 |
| Q2-3, early | 1963-1988 | May 1963-April 1989 |
| Q2-3, recent | 1990-2023 | May 1990-April 2024 |

The shared extended stock/index cache ends in December 2024; accounting history starts in 1957. Under the implemented GS rule, April 1962 formation needs accounts beginning in 1956. An updated cache through December 2025 would permit the final complete annual holding window May 2024-April 2025. A partial May-December 2025 period should not be presented as a full annual paper-style return. For requirements saying "present," use the latest available complete observations and identify the extraction date; Q2-3 explicitly requests the calendar endpoint 2025.

The early/recent LSV comparison deliberately retains only complete annual windows wholly inside each requested calendar period. It consequently excludes May 1989-April 1990 from both sides. This is transparent and avoids extending a calendar subsample with future returns, but it is not an exhaustive partition of the full holding-return series. If HW1 intends a partition of formation cohorts, April 1989 should instead be assigned to the early formation sample with its later follow-up labelled explicitly. The chosen interpretation should be stated and reconciled with the assignment; the paper's annual definitions do not justify silently treating different window definitions as equivalent.

## 2. JT original replication is strong, with limits on Tables II/VII

For the selected original sample:

- All 96 Table I portfolio means and t-statistics match the frozen supplied-notebook output to floating-point precision. The maximum absolute return discrepancy is below 1e-12 basis points per month.
- Across the 32 winner-minus-loser strategies, the mean absolute difference from the published paper is 5.75 basis points per month. This is a different comparison from notebook parity.
- Panel A (6,6) returns 0.8794% per month, versus 0.9500% published, with conventional t = 2.906 versus 3.07. Panel B gives 1.0215% versus 1.1000%, t = 3.447 versus 3.61.

Tables II and VII have complete numerical output and published comparisons, but their implementation files and run records still identify them as drafts. Table II uses monthly post-ranking OLS beta and market capitalization fixed at the month before each cohort forms. These are reasonable explicit conventions; the paper does not specify all of their details. The spread beta is -0.1302 versus -0.08 published. Loser and winner mean capitalizations are approximately $173.11m and $356.05m, versus $208.24m and $495.13m published. These gaps deserve a concrete discussion of size timing, weights, availability, and vintage; they cannot be attributed to vintage alone from the current evidence.

Table VII uses 265 balanced calendar cohorts: January 1965-January 1987 formations with all 36 follow-up months ending by December 1989. This is a calendar cutoff, not a future-survival requirement imposed on each stock. Its first-year cumulative spread is 8.6434%, and its 36-month cumulative spread is 3.8794%. It reproduces the broad first-year momentum and subsequent decline. The complete-cohort boundary and Newey-West bandwidth h-1 remain explicit implementation choices rather than independently verified original-program choices. Summing event-month zero-cost returns and monthly equal-weight rebalancing are consistent with the documented construction; these are not errors merely because a buy-and-hold alternative exists.

Historical numerical identity with a 1993 paper is not necessary for a credible replication using revised data. The outstanding issue is explaining economically meaningful gaps and distinguishing verified notebook parity from a verified original-paper implementation.

## 3. LSV Table V is a substantive incomplete replication

All 64 requested positions are displayed, but only 54 have finite calculated values in the original sample. Ten earnings/cash-flow growth statistics are undefined because at least one required aggregate level is nonpositive. They have finite published counterparts. The extended cases contain 13 undefined cells for NYSE, 16 for all exchanges, four in the early sample, and 16 in the recent sample.

The strongest warning comes before the growth calculation:

| Original Table V statistic | Calculated | Published |
| --- | ---: | ---: |
| B/M glamour portfolio mean B/M | -0.0477 | 0.225 |
| B/M glamour portfolio mean E/P | -0.1983 | 0.029 |
| B/M value portfolio mean B/M | 2.6306 | 1.998 |
| B/M value portfolio mean E/P | -0.0353 | 0.004 |

The baseline retains negative book equity in B/M sorts to follow the supplied notebook. There are 457 negative-B/M observations among 4,515 original-sample B/M-glamour stock-formations, or 10.12%. Their proportion reaches 41.45% in one formation year. Extreme distressed firms have very large negative ratios: the minimum B/M is approximately -143.32 and minimum E/P approximately -122.48. For example, two USD observations alone contribute about -0.0502 to the final glamour E/P average under the implemented equal-cohort weighting. The discrepancy is therefore observable in the constituents; it is not just display rounding or a missing HTML value.

The paper's stated positive-ratio exclusion applies to C/P and E/P portfolio classification. It is insufficient evidence to assert that the paper explicitly required a positive-B/M filter. Do not add a filter solely to force the table to match. Investigate the historical universe, CEQ/book-equity definition, link and security selection, fiscal/accounting timing, and economically extreme observations, then present justified sensitivities. A positive-book-equity sensitivity can be informative if separately labelled. Preserve the supplied-notebook baseline for comparison.

The code correctly distinguishes a ratio of average portfolio accounting levels from an average of firm growth rates, and does not fabricate geometric growth from negative levels. Those safeguards are strengths. They explain why dashes appear; they do not make Table V a completed numerical replication.

## 4. LSV Tables IV and VI need a stronger discrepancy assessment

LSV Tables I and II reproduce the central value-return patterns, but are not numerically identical to the paper. Across R1-R5, AR, and SAAR, mean absolute cell differences are 1.53 percentage points in Table I and 1.78 percentage points in Table II. The original Table I AR value-minus-glamour spreads are 11.22% versus 10.5% for B/M, 12.15% versus 11.0% for C/P, 8.33% versus 7.6% for E/P, and 6.58% versus 6.8% for GS. Cumulative CR5 cells naturally amplify annual differences and should be assessed separately from annual returns. These are credible qualitative replications, with remaining numerical differences.

Table IV contains all nine specifications and all 32 coefficients, uses annual cross-sectional OLS, and computes inference from annual coefficient variation. The positive-part and negative-dummy implementation is consistent with the displayed specifications. However, some published conclusions are not reproduced:

- Univariate E/P+: coefficient 0.2377, t = 1.568, versus 0.526, t = 2.541 in the paper.
- Full E/P specification: E/P+ coefficient 0.1108, t = 0.827, versus 0.394, t = 2.008.
- Univariate C/P+: coefficient 0.1595, t = 2.765, versus 0.356, t = 4.240.

The E/P findings are changes in conventional significance, not just differences in the last displayed digit. Discuss them expressly rather than presenting only the paper's qualitative interpretation. The absence of winsorization, extreme ratio observations, changing model-specific complete-case samples, and accounting conventions are candidates to investigate; the present audit does not establish which one explains the gaps.

Table VI's one-year mean spreads are close to the paper: 7.853% versus 7.9% for C/P tails, 10.586% versus 10.2% for C/P-GS, and 6.461% versus 6.3% for B/M tails. Its five-year mean C/P and B/M tail spreads also closely match. Inference is a separate concern:

| Five-year strategy | Calculated rectangular HH t | Published t | Bartlett/Newey-West sensitivity t |
| --- | ---: | ---: | ---: |
| C/P tails | 22.775 | 7.630 | 7.028 |
| C/P-GS | 9.285 | 5.939 | 5.922 |
| B/M tails | 23.107 | 7.104 | 5.488 |

The sensitivity uses the same 18 cohort returns, four lags, and n/(n-1) correction. It changes only the lag weights. The implemented rectangular Hansen-Hodrick formula is internally consistent, and the paper names Hansen-Hodrick. This audit does not establish a coding error or justify replacing it merely because another estimator is closer to printed t-statistics. It does establish considerable small-sample inference sensitivity and unresolved published-inference differences. Report the sensitivity, examine the covariance convention, and avoid overstating the precision of t-statistics around 23 from 18 heavily overlapping cohorts. Keep the documented B/M caption-versus-heading ambiguity and its alternative definition visible.

Figure 2 is present for the original and extended cases, and the saved-output checks verify its return series against Table VI. Recession/decline marker conventions are disclosed. These components are not missing deliverables.

## 5. The substantive answers are mostly already present

**Recent momentum:** the recent NYSE/AMEX (6,6) spread remains positive: Panel A 0.5369% monthly, conventional t = 1.424; Panel B 0.5877%, t = 1.585. No recent Table I winner-minus-loser strategy reaches the conventional two-sided 5% threshold among the 32 tested, versus 31 in the early selected run. The correct statement is weaker evidence of momentum in this universe/window, with positive means; it is not proof that momentum is zero. This analysis must be rerun for the completed required window before it is treated as the final HW1 comparison.

**FF3 alpha:** the original-sample Panel A (6,6) estimate is 1.3331% per month, HAC t = 5.6317, two-sided asymptotic p = 1.784e-8, 95% interval [0.8691%, 1.7970%], N = 300, R-squared = 0.1628. Regressing the zero-cost spread directly on Mkt-RF, SMB, and HML is appropriate; RF cancels between the legs. An independent matrix-kernel covariance calculation reproduces the saved five-lag result. Twelve lags give t = 5.5183, with the same point estimate. Q1-5 is completed and its conclusion is robust to this bandwidth check. HW1 does not require every raw-return mean to be smaller than alpha.

**Recent value:** Table I AR decile spreads fall for all four signals: B/M 10.49 to 9.24, C/P 11.79 to 9.13, E/P 8.41 to 5.35, and GS 4.78 to 2.75 percentage points. SAAR spreads fall for three of four. Table VI annual tail spreads are mixed: C/P tails 8.34% to 7.09%, C/P-GS 9.31% to 9.09%, and B/M tails 7.22% to 8.91%. Recent conventional annual t-statistics remain about 2.03-2.17 for these three strategies. The submission appropriately says that weakening depends on the portfolio/return measure and that positive value premiums remain. A formal early/recent mean-difference test would strengthen the answer, but HW1 does not explicitly demand one. A lower t-statistic by itself does not establish a smaller mean.

## 6. Reproducibility, checks, and audit limits

Checks performed for this audit:

1. Ran the complete offline test suite: **154 tests passed**. The log is `tmp/audit_unittest.log`.
2. Validated the selected saved-output structures and consistency checks for the original and four extended LSV cases, and all five selected JT cases.
3. Independently recomputed all selected JT Table I monthly means/conventional t-statistics, winner-minus-loser identities, and agreement of Table II's (6,6) spread with Table I.
4. Compared the original outputs with the separate published benchmark CSVs and inspected the supplied paper's relevant methodology/table pages, including visual inspection of LSV Tables IV, V, and VI.
5. Independently reproduced FF3 OLS and five-lag HAC standard errors using a matrix of Bartlett time weights; checked a twelve-lag sensitivity.
6. Recomputed Table VI HH and Bartlett sensitivity statistics directly from the saved cohort returns, and examined original B/M-glamour constituents.
7. Confirmed that the current LSV HTML hash matches its saved submission validation record. That record reports 25 source tables, 15 embedded figures, 79 table panels, and no JavaScript errors for the checked presentation.

The saved LSV notebook replay reports all 320 cells matching within 2.22e-16 decimal-return units, plus checks of formation characteristics, membership, annual stock returns, and cohort returns. This audit verified that its summary, notebook, and all three raw-input hashes still match the current files. This is strong evidence of reference-program reproducibility. Matching an instructional notebook and passing constructed-data tests do not establish full compliance with requested dates, exact historical-data identity, or correct original-paper interpretation for every newly implemented table.

This was an audit, not a data refresh or a full portfolio recomputation from raw stock files. Source data, replication code, and submitted results were not edited. Temporary audit extracts and this report were added. No new WRDS data were downloaded. The saved reference replay was inspected rather than rerun in full.

## 7. Recommended order of work before submission

1. **Complete the requested samples.** Obtain suitable accounting warm-up and updated stock/index coverage; calculate full-window JT results for both required exchange universes; extend/rebuild the relevant JT and LSV subperiod results. Keep requested dates distinct from effective dates. Do not splice Table I means or other summaries to simulate full-sample estimates.
2. **Resolve the LSV Table V sample discrepancy.** Inspect negative-equity/extreme-ratio constituents and justified sample/book-equity sensitivities; trace the nonpositive aggregate levels behind the ten original missing statistics. If a discrepancy remains, make the limitation and its economic implications explicit rather than claiming full replication.
3. **Investigate material original-paper differences.** Prioritize Table IV's E/P findings, Table VI multi-year standard errors, and JT Table II size aggregation/Table VII endpoints. Separate established errors, data-vintage differences, and underidentified historical implementation choices.
4. **Reconcile LSV's subsample definition with the intended partition.** State whether the split refers to formation cohorts or strictly bounded calendar returns and account for the annual window crossing 1989/1990.
5. **Refresh the written interpretation after the reruns.** The existing recent-momentum, FF3, and mixed value-weakening discussions provide a good foundation. Formal early/recent difference tests and further bandwidth checks are useful enhancements, not invented assignment requirements.

Two smaller documentation improvements: list the Compustat extraction-population filters alongside the economic filters; correct the Table IV note claiming that SIZE slopes depend on whether market equity is measured in dollars or millions. With log size and an intercept, changing currency units shifts the intercept, not the size slope. The original LSV baseline also contains 403 CAD formation observations combined with USD market equity. The inconsistency is disclosed and a separate USD-only Table I sensitivity exists; it should be discussed as a limitation rather than silently treated as a fully harmonized ratio dataset.

The current work is credible as an explicitly limited replication package. It should not yet be described as satisfying every HW1 requirement.
