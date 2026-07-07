"""Minimal end-to-end example on a synthetic Einstein-mode spectrum."""
import numpy as np

from elphgap import moments, tc_allen_dynes, tc_eliashberg

# synthetic alpha^2F: single Einstein mode at 60 meV carrying lambda = 1
w = np.linspace(1.0, 120.0, 2000)
g = np.exp(-0.5 * ((w - 60.0) / 1.5) ** 2)
a2f = g / (2.0 * np.trapezoid(g / w, w))  # normalized so 2*int(a2F/w) = 1

lam, wlog, wsq = moments(w, a2f)
result = tc_eliashberg(w, a2f, mu_star=0.13)
print(f"lambda={lam:.3f}  omega_log={wlog:.1f} meV")
print(f"Tc Allen-Dynes (mu*=0.13): {tc_allen_dynes(lam, wlog, wsq, mu_star=0.13):.1f} K")
print(f"Tc isotropic ME (mu*=0.13): {result.tc_kelvin:.1f} K")
