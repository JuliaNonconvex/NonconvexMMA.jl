# NEWS

## 1.3.6

- Docs: the logo now appears only in the sidebar, not in the page body.
## 1.3.5

- Remove the use of the `debugging` flag removed in NonconvexCore 2.3 (the
  debug `@show` branch and the test toggling it are dropped).
## 1.3.4

- Docs: package-name section titles and left-aligned logos (no code changes).
## 1.3.3

- Fixed `auto_scale` destroying the objective on unconstrained models:
  without constraints the objective/constraint norm ratios are empty and the
  objective was scaled by zero (the reported minimum became NaN). The
  scaling is now skipped in that case; added a regression test.

## 1.3.2

- Validated the MMA kwargs and fixed trace and maxiter bookkeeping;
  informative README and QA parity (docs badges, Aqua/JET, formatting);
  codecov OIDC/v5 uploads and the organization logo.

## 1.3.1

- Fixed docs references and the scalar-rho constructor dispatch, with
  coverage tests.

## 1.3.0

- NonconvexCore 2 is now required with strict solver capability
  declarations (`infinite_bounds` declared); the local `[sources]` test
  override was dropped.
- DocumenterFragments docs, tests migrated to TestItems, JET/Aqua/formatting
  quality gates, and Julia 1.10+.

## 1.2.2

- Documented that `MMA02` implements the globally convergent MMA (GCMMA)
  from Svanberg's 2002 paper: the `GCMMA` alias now carries a docstring, and
  the README explains the difference between MMA87 and MMA02/GCMMA.
