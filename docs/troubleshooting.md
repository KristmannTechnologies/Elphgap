# Troubleshooting

Exit codes: **0** ok · **2** broken input · **3** Tc censored · **4** invalid
user choice · **5** solver error. Check with `echo $?` right after the command.
Errors print as `elphgap: error[<code>]: <message>`; the `<code>` is stable
(the same string appears in the `--json` `warnings[].code`).

## `error[format_undetectable]` (exit 4)

Auto-detection uses only decisive signatures — the QE header `frequencies in
Rydberg` or an EPW footer line (`Phonon smearing (meV)`, `Integrated el-ph
coupling`, `Summed el-ph coupling`, `Fermi window`). There is **no** frequency-
magnitude guess. If your file has neither signature, pass `--format qe|epw`
explicitly. (This is deliberate: a magnitude guess would silently rescale a
legitimate low-energy meV spectrum by 13605×.)

## Units mix-up — ω_log and Tc off by ~13605× (NOT λ)

If you force the wrong `--format`, the frequency axis is mis-scaled by the
Rydberg→meV factor (13605.693). What moves and what does not:

- **λ does not change.** λ = 2∫α²F/ω dω is dimensionless and invariant under a
  consistent rescaling of the whole ω-axis, so a units error is invisible in λ.
- **ω_log, ω_2, and Tc DO change** (they carry energy units) — by ~13605× or
  its inverse. That is the red flag.

QE `a2F.dos` frequencies are in **Rydberg** (phonons a few ×10⁻³ Ry); EPW
`prefix.a2f` frequencies are in **meV** (tens of meV). If `inspect` shows
ω_max in the hundreds–thousands of meV, or ω_log/Tc absurdly large or tiny,
you probably read Ry as meV (or vice versa). `inspect` prints `format`,
`units_in`, and the ω range — verify them.

## `error[epw_column_count]` (exit 2)

An EPW `prefix.a2f` must have exactly `1 + 2N` columns: ω, then `N` a2F columns
(one per phonon smearing), then `N` cumulative-λ(ω) columns. A different count
(e.g. an even number, or a hand-trimmed file) is rejected rather than guessed.
Run `elphgap inspect FILE --format epw` on the untrimmed EPW output.

## `error[column_is_lambda]` / `error[column_out_of_range]` (exit 4)

You selected a column that is not a2F. In an EPW file columns `N+2 … 2N+1` are
**cumulative λ(ω)**, not a2F — feeding them to the solver would be nonsense, so
they are rejected. `inspect` lists the valid a2F columns and each one's λ.

## `error[column_required]` (exit 4)

The EPW file has several a2F smearing columns and `tc` has no canonical default
(neither the first nor the last smearing is universally "the" answer). Look at
`inspect` (it shows every smearing and its λ), decide which is converged, and
pass `--column N`.

## `error[negative_a2f]` (exit 2)

The selected a2F column has negative values — usually numerical noise or an
unconverged double-delta integration. By default this is refused, not silently
clamped. If (and only if) the negatives are small rounding noise, pass
`--clamp-negative` to clamp them to 0; the output then reports the count and the
most-negative value so you can confirm it really was noise. Large negatives mean
the α²F is under-converged — fix it upstream.

## `error[non_finite]` (exit 2)

The data block contains `NaN`/`Inf`. elphgap refuses to guess; regenerate the
α²F.

## `error[malformed_row]` (exit 2)

A line that starts with a number contains a non-numeric token — a units tag,
stray text, or a Fortran overflow (`********`). Open the file at the reported
`path:line`. Comment lines (`#`, `!`) and pure-text footers (`lambda = 1.05`,
`Phonon smearing (meV)`) are skipped automatically.

## Dropped / clamped warnings (not errors)

- `dropped_nonpositive`: EPW files start at ω = 0, where the a2F/ω moments are
  singular. That row is dropped and reported. Normal.
- `dropped_below_clip`: `--clip-below MEV` removed a low-ω tail (as requested).
- `lambda_crosscheck`: for EPW, `2∫α²F/ω` disagreed with the file's cumulative-λ
  column by >5% — often a sign the a2F/λ column mapping is not the standard
  layout, or the grid was heavily clipped. Check the file against `inspect`.

## `Tc censored` / exit 3

Tc fell below the floor set by `--n-max`. Either the coupling is too weak for a
finite Tc at this μ\*, or the cap is too tight. The default `--n-max` is 4096;
`--fast` uses 512 (a higher floor). Raise `--n-max` and re-run. The floor scales
as ω_c / (2π·n_max); the output prints `rho(T_floor)` so you can see how far
below 1 it is. See [limitations.md](limitations.md).

## Parameter errors (exit 4)

Valid ranges: `--mu-star` in [0, 1) (physical 0.10–0.16), `--cutoff-factor` > 0,
`--n-max` ≥ 4, `--clip-below` ≥ 0, `--column` ≥ 2 and pointing at an a2F column.

## `error[solver_no_bracket]` (exit 5)

The Tc bisection could not bracket a root below `t_max` (2000 K) — typically an
extreme/unphysical λ or ω. Sanity-check the input spectrum (`inspect`).

## `elphgap: command not found`

The console script is not on your PATH (installed into an inactive venv, or via
`pip install --user`). Fall back to `python -m elphgap ...`. To locate the
shipped example after install:
`python -c "import elphgap; print(elphgap.example_a2f_path())"`.
