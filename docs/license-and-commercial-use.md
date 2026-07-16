# License and commercial use

elphgap is licensed under **GNU General Public License v3.0 or later
(GPL-3.0-or-later)** — the same copyleft license as Quantum ESPRESSO and EPW.
This page is a plain-language FAQ. **It is not legal advice**; the
[`LICENSE`](../LICENSE) text is what actually governs, and it comes with no
warranty and no liability. If in doubt, consult a lawyer.

## Can I use it commercially?

Yes. The GPL does not restrict *use*. You may run elphgap for any purpose —
academic, commercial R&D, for-profit consulting — with no obligation to share
anything, as long as you are only *running* it, not *distributing* it.

## Do I own the numbers it computes?

Yes. The Tc, λ, ω_log, gaps, and JSON reports elphgap produces from **your**
α²F input are your results. Running a GPL program over your data does not place
the *output* under the GPL — the license covers the program's source code, not
the data it computes. Publish and commercialize your results freely (and cite
the code and the methods — see the [README](../README.md#cite)).

## Can I run it as an internal tool or a paid service?

Yes. GPL-3.0 (unlike AGPL-3.0) has **no network-use clause**. Offering elphgap
behind an API or as part of a hosted service, internal or paid, does not by
itself trigger any obligation to distribute source. You are running it, not
conveying it.

## When does copyleft actually kick in?

When you **distribute (convey)** elphgap or a work based on it. Specifically:

- Redistributing elphgap, or a modified version, to anyone else → you must offer
  the complete corresponding source under GPL-3.0-or-later, keep the license and
  copyright notices, and state your changes.
- Shipping a program that **incorporates elphgap** — e.g. `import elphgap` in an
  application you distribute, or bundling it into a product — makes a *combined
  work*. On distribution, the whole combined work must be offered under
  GPL-compatible terms (GPL-3.0-or-later for the copyleft parts). This is the
  "embedding limit": you cannot fold elphgap into a **proprietary, distributed**
  product and keep that product closed.
- Purely internal use, and computing results you then publish, are **not**
  distribution and carry no such obligation.

## I want to embed it in closed, distributed software

That is exactly the case GPL copyleft restricts. If you need elphgap under terms
other than the GPL (for example, to link it into a proprietary product you
distribute), a separate commercial/dual license may be possible.

**Dual-licensing enquiries:** contact **Kristmann Technologies** (the release
steward for this project; see the repository owner on GitHub). Please describe
your intended use. Note this is a placeholder contact route — no dual-licensing
offer is guaranteed.

## Quick reference

| What you do | GPL obligation |
|---|---|
| Run it, compute Tc for your paper/product | None (own your results) |
| Modify it, keep the changes to yourself | None |
| Offer it as a hosted/paid API service | None (GPLv3 has no network clause) |
| Distribute it or a modified copy | Offer complete source, GPL-3.0-or-later |
| `import elphgap` into software you distribute | Combined work → GPL-compatible on distribution |
| Embed in a closed, distributed product | Not allowed under GPL — ask about dual licensing |

Again: summary only, not legal advice. The [`LICENSE`](../LICENSE) file controls.
