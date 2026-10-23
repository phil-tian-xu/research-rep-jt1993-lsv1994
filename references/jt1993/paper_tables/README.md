# Published JT1993 table values

Source: Jegadeesh, N., and S. Titman (1993), "Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency," Journal of Finance
48(1), 65-91. The supplied PDF is under `docs/raw_paper/`.

These files contain published paper values, independently of the WRDS notebook
outputs and of this project's replication results. The sample is NYSE/AMEX,
January 1965-December 1989. Do not replace these values with a new analysis run.

| File | Printed page | Keys | Contents |
| --- | --- | --- | --- |
| `table1.csv` | 70 | panel, J, K, leg | All 96 monthly means and t-statistics |
| `table2.csv` | 73 | portfolio | 11 betas and 10 average capitalizations |
| `table7.csv` | 84 | event_month | 36 monthly/cumulative returns and t-statistics |

Returns are decimals, t-statistics and betas are unitless, and market
capitalization is in USD millions. Preserve four decimal places for published
returns and two for t-statistics, betas, and capitalization. A blank spread
capitalization in Table II represents the paper's dash, not zero.

Table I Panel A has no gap. Panel B skips one week. `loser`, `winner`, and
`buy_sell` correspond to Sell, Buy, and Buy-sell. The Table I values were copied
from the already checked `paper_mean`/`paper_t` columns in
`references/jt1993_table1_reference_results.csv`; the notebook's `reference_*`
columns are deliberately excluded. Tables II and VII were transcribed and
checked against the supplied paper images.

Keep the published cumulative returns in Table VII as printed. Do not rebuild
them by summing the rounded monthly values: rounding differences are present
(for example, event month 3). The paper uses autocorrelation-consistent
Newey-West standard errors for cumulative returns; it does not specify the
bandwidth or finite-sample correction used by our draft implementation.
