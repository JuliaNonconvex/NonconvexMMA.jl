# NEWS

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
