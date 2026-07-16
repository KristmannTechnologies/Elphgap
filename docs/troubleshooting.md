# Troubleshooting

Exit codes: **0** ok · **2** parse/format error · **3** Tc censored · **4**
invalid parameters. Check with `echo $?` right after the command.

## "parse error: ... expected a numeric data row" (exit 2)

A line that starts with a number contains a non-numeric token — usually a units
tag or stray text glued onto a data row, or a real column that is `********`
(a Fortran overflow). Open the file at the reported `path:line`. Comment lines
(`#`, `!`) and pure-text footers (e.g. `lambda = 1.05`) are skipped
automatically; only lines that *look* like data but aren't will trip this.

## "requested a2F column N but the file has only M column(s)" (exit 2)

You passed `--column N` beyond the file width. Run `elphgap inspect FILE` first —
it prints `column: X of M`, so you can see how many α²F columns exist.

## Wrong format detected / λ off by ~13600×

Autodetection keys on header strings ("frequencies in Rydberg" → `qe`, "meV" →
`epw`) and falls back to the frequency magnitude. If a file has no informative
header, it can be misread. Symptoms and fix:

- **λ or ω_log absurdly large or small, ω_max in the thousands of meV**: the
  Rydberg→meV conversion (×13605.693) was applied to a file already in meV, or
  vice versa. Force it: `--format qe` (frequencies in Ry) or `--format epw`
  (meV). `inspect` prints the detected format and `units_in`; verify it.
- QE `a2F.dos` frequencies are in **Rydberg** (a few ×10⁻³ Ry for phonons);
  EPW `prefix.a2f` frequencies are in **meV** (tens of meV). ω_max around
  100–2000 meV is a red flag that Ry was read as meV.

## Warnings about dropped rows / clamped α²F (not an error)

- `dropped N row(s) with omega <= 0 meV`: EPW files start at ω = 0, where the
  `a2F/ω` moments are singular. That row is discarded and reported. Normal.
- `clamped N negative a2F value(s) to 0`: numerical noise or an unconverged
  double-delta integration produced α²F < 0. Values are clamped and counted.
  Many such rows means the α²F itself is under-converged — fix it upstream.
- To cut a noisy low-ω tail explicitly, use `--clip-below MEV` (e.g.
  `--clip-below 1.0` drops ω ≤ 1 meV). This is reported too.

## "EPW file has K a2F columns ... using column 2" (not an error)

EPW writes α²F at several phonon smearing values. elphgap defaults to the first
smearing and warns because the choice is physics. Inspect the columns, decide
which smearing is converged, and pass `--column N` to silence the warning and
pin your choice.

## "Tc censored" / exit code 3

The Tc fell below the floor set by `--n-max`. Either the coupling is too weak
for a finite Tc at this μ\*, or the cap is too tight. Raise it, e.g.
`--n-max 2048`, and re-run. The floor scales as ω_c / (2π·n_max); the message
prints `rho(T_floor)` so you can see how far below 1 it is. See
[limitations.md](limitations.md#censoring-semantics-tc-exit-code-3).

## "--mu-star must be in [0, 1)" and friends (exit 4)

Parameter out of range. Valid: `--mu-star` in [0, 1) (physical values
0.10–0.16), `--cutoff-factor` > 0, `--n-max` ≥ 4, `--column` ≥ 2.

## `elphgap: command not found`

The console script is not on your PATH — the wheel was installed into a venv
that is not active, or you used `pip install --user`. Fall back to
`python -m elphgap ...`, which works from anywhere the package is importable.
