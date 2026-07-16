# License and commercial use

elphgap is licensed under **GNU General Public License v3.0 or later
(GPL-3.0-or-later)** — the same copyleft **family** as Quantum ESPRESSO and EPW,
which are themselves distributed under **GPLv2** (a different version of the same
family, not the identical license). This page is a plain-language FAQ. **It is
not legal advice**; the [`LICENSE`](../LICENSE) text is what actually governs,
and it comes with no warranty and no liability. If in doubt, consult a lawyer.

## Can I use it commercially?

Yes. The GPL does not restrict *use*. You may run elphgap for any purpose —
academic, commercial R&D, for-profit consulting — with no obligation to share
anything, as long as you are only *running* it, not *distributing* it.

## Who owns the numbers it computes?

In the ordinary case, you do. Running a GPL program over **your** α²F input to
produce Tc, λ, ω_log, gaps, and JSON reports is use, not distribution, and such
**ordinary numerical output is typically not covered by the GPL** — the license
covers the program's source, not the data it computes. The FSF notes an
edge case: output *can* be restricted if it itself contains substantial
protected parts of the program (not the situation for numeric results here).
Whether any given output crosses that line is content- and jurisdiction-
dependent, so treat this as the general rule, not a guarantee for every case.
See the [GNU GPL FAQ on program output](https://www.gnu.org/licenses/gpl-faq.en.html#WhatCaseIsOutputGPL).
Subject to that, publish and commercialize your results freely (and cite the
code and methods — see the [README](../README.md#cite)).

## Can I run it as an internal tool or a paid service?

Yes. GPL-3.0 (unlike AGPL-3.0) has **no network-use clause**. Offering elphgap
behind an API or as part of a hosted service, internal or paid, does not by
itself trigger any obligation to distribute source. You are running it, not
conveying it.

## When does copyleft actually kick in?

When you **distribute (convey)** elphgap or a work based on it:

- Redistributing elphgap, or a modified version, to anyone else → you must offer
  the complete corresponding source under GPL-3.0-or-later, keep the license and
  copyright notices, and state your changes.
- Shipping a program that **incorporates elphgap** — e.g. `import elphgap` in an
  application you distribute, or bundling it into a product. The FSF's position
  is that combining GPL code with other code into one program generally makes a
  *combined work* whose whole must, on distribution, be offered under
  GPL-compatible terms. Where the boundary of a "combined work" falls (linking,
  separate processes, mere aggregation) is a fact-specific legal question — see
  the [GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.en.html). The practical
  takeaway: do not assume you can fold elphgap into a **proprietary, distributed**
  product and keep that product closed.
- Purely internal use, and computing results you then publish, are **not**
  distribution and carry no such obligation.

## I want to embed it in closed, distributed software

That is the case GPL copyleft is designed to restrict, and it is a legal
question specific to your situation — consult counsel. **elphgap is offered
under GPL-3.0-or-later only; there is no alternative or dual license available
at this time.**

## Quick reference

| What you do | GPL obligation |
|---|---|
| Run it, compute Tc for your paper/product | None; ordinary numeric output typically not GPL-covered |
| Modify it, keep the changes to yourself | None |
| Offer it as a hosted/paid API service | None (GPLv3 has no network clause) |
| Distribute it or a modified copy | Offer complete source, GPL-3.0-or-later |
| `import elphgap` into software you distribute | Generally a combined work → GPL-compatible on distribution (fact-specific) |
| Embed in a closed, distributed product | Restricted by the GPL; GPL-only, no alternative license offered — consult counsel |

Again: summary only, not legal advice. The [`LICENSE`](../LICENSE) file controls.
