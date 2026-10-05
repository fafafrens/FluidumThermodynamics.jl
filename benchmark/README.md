# FluidumEOS thermodynamic benchmarks

Run from the package root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl
julia --project=. benchmark/accuracy.jl
```

BenchmarkTools and MuladdMacro are benchmark-only dependencies.
`baseline.jl` preserves the package kernel from before these optimizations.
`legacy_fluidum.jl` preserves the original Fluidum.jl numerical code from
commit `4fdd25fc311d64619d8b2f2af44054775dbec50f`, including its
`@fastmath` and `@muladd` annotations. The original kernel is loaded in
isolation; these measurements do not include loading the full Fluidum package.

Scalar inputs are dereferenced through `Ref` to prevent constant folding.
Scalar measurements use 1,000 evaluations per sample to resolve timings below
the timer's single-call resolution. The sweep evaluates 4,096 different
temperatures from 0.08 to 1.0 GeV and writes pressure, gradient, and Hessian
into a preallocated output vector. Sweep timings are divided by 4,096.
Compilation is excluded. The default sampling budget is one second per case;
set `FLUIDUM_BENCH_SECONDS` to change it.

## Recorded results

Measured 2026-10-05 with Julia 1.13.0, CPU target `apple-m4`, one Julia thread.
All cases below use Float64 and GeV/fm³ for a direct comparison. Timings depend
on hardware, Julia/compiler versions, and system load.

| Kernel | Single T=0.2 GeV, median ns | Sweep, median ns/evaluation | Sweep, minimum ns/evaluation | Allocations |
|---|---:|---:|---:|---:|
| Previous package kernel | 8.71 | 7.73 | 7.69 | 0 |
| Original Fluidum kernel | 8.21 | 6.76 | 6.63 | 0 |
| Optimized package kernel | 6.04 | 4.83 | 4.77 | 0 |

The optimized sweep is approximately 1.60× faster than the previous package
kernel and 1.40× faster than the original Fluidum kernel. Default GeV⁴
evaluation also measured 4.83 ns/evaluation in the sweep.

The changes reuse temperature and denominator reciprocals, combine shared
exponential/temperature factors, and use explicit `muladd` operations.
The package kernel does not use `@fastmath` or an approximate exponential.

## Numerical validation

All 761 regression assertions pass. The accuracy script compares Float64
pressure and analytic derivatives against 256-bit pressure and ForwardDiff
derivatives at 1,001 temperatures over 0.08–1.0 GeV, in NaturalGeV, GeVFm,
and MeVFm conventions.

| Quantity | Maximum relative error |
|---|---:|
| Pressure | 4.40e-15 |
| Gradient | 6.81e-15 |
| Hessian | 1.41e-14 |

This measures numerical agreement for the implemented fit over that interval,
not the fit's physical accuracy.
