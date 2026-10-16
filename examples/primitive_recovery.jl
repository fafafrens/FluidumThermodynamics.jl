# Conserved → primitive recovery for a (T, μ) fluid with any one-charge EOS.
#
# The generic 2×2 damped Newton of a finite-volume solver: with q ≡ E + p = wγ²
# and r ≡ √(q² − M²), the energy and charge closures are
#
#     fe = w q − r²,      fn = n q − N r,
#
# and their Jacobian needs only p, s = ∂p/∂T, n = ∂p/∂μ and the pressure Hessian
# at the candidate (T, μ) — one `thermodynamic` evaluation. The energy
# derivatives follow from Euler, e_T = T p_TT + μ p_Tμ and e_μ = T p_Tμ + μ p_μμ.
# Nothing here depends on the EOS beyond that evaluation, so the same loop
# serves IdealQCD, HeavyQuark (charm as the conserved charge), a HadronResonanceGas
# or a composed EOS. Quantities are in the EOS convention (NaturalGeV by default).
#
# Run with `julia --project=. examples/primitive_recovery.jl`; including the file
# only defines the functions.

using FluidumThermodynamics, StaticArrays

"""
    conserved_from_primitive(eos, T, μ, vx, vy) -> (N, Mx, My, E)

Lab-frame charge density, momentum and energy of a fluid at rest-frame `(T, μ)`
moving with velocity `(vx, vy)`.
"""
function conserved_from_primitive(eos, T, μ, vx, vy)
    x = GrandCanonicalPoint(T, μ)
    th = thermodynamic(eos, x)
    w = enthalpy(x, th)
    v2 = vx^2 + vy^2
    v2 < 1 || throw(DomainError(v2, "superluminal velocity"))
    γ2 = 1/(1 - v2)
    (charge_density(th, 1)*sqrt(γ2), vx*w*γ2, vy*w*γ2, w*γ2 - th.pressure)
end

# Residual (fe, fn) and its Jacobian at a candidate (T, μ).
@inline function residual_jacobian(eos, T, μ, E, M2, N)
    x = GrandCanonicalPoint(T, μ)
    th = thermodynamic(eos, x)
    p, s, n = th.pressure, entropy_density(th), charge_density(th, 1)
    pTT, pTμ, pμμ = th.hessian[1,1], th.hessian[1,2], th.hessian[2,2]
    w = enthalpy(x, th)
    q = E + p
    r = sqrt(q*q - M2)
    fe = w*q - r*r
    fn = n*q - N*r
    c = n - N*q/r
    eT = T*pTT + μ*pTμ
    eμ = T*pTμ + μ*pμμ
    f = SVector(fe, fn)
    J = SMatrix{2,2}(q*eT + (w - q)*s, q*pTμ + c*s, q*eμ + (w - q)*n, q*pμμ + c*n)
    f, J
end

# Damped Newton on (T, μ): the step is capped to `maxfrac` of |u| so cold starts
# stay physical (q² > M²); near the root the cap is inactive. Stops when
# ‖δ‖ ≤ atol + rtol‖u‖. Returns the solution and the iteration count.
@inline function recover(eos, E, M2, N, T0, μ0; rtol=1e-12, atol=1e-14, maxit=50, maxfrac=1.0)
    u = SVector(T0, μ0)
    for it in 1:maxit
        f, J = residual_jacobian(eos, u[1], u[2], E, M2, N)
        δ = J \ f
        nrm = abs(δ[1]) + abs(δ[2])
        unorm = abs(u[1]) + abs(u[2])
        cap = maxfrac*unorm + atol
        α = ifelse(nrm > cap, cap/nrm, one(nrm))
        u = u - α*δ
        nrm < atol + rtol*unorm && return u, it
    end
    u, maxit
end

"""
    primitive_from_conserved(eos, N, Mx, My, E; T0, μ0) -> (T, μ, vx, vy), iterations

Recover the rest-frame `(T, μ)` and the velocity from the conserved variables.
Seed `T0, μ0` with the cell's previous primitives when available.
"""
function primitive_from_conserved(eos, N, Mx, My, E; T0, μ0)
    M2 = Mx*Mx + My*My
    u, it = recover(eos, E, M2, N, T0, μ0)
    T, μ = u[1], u[2]
    q = E + pressure(eos, GrandCanonicalPoint(T, μ))
    SVector(T, μ, Mx/q, My/q), it
end

# Timing through a function, so that nothing is a global of the script.
function demo(name, eos, T, μ, vx, vy; samples=100_000)
    N, Mx, My, E = conserved_from_primitive(eos, T, μ, vx, vy)
    cold, itc = primitive_from_conserved(eos, N, Mx, My, E; T0=0.15, μ0=0.0)
    warm, itw = primitive_from_conserved(eos, N, Mx, My, E; T0=1.01T, μ0=1.05μ)
    allocs = @allocated primitive_from_conserved(eos, N, Mx, My, E; T0=1.01T, μ0=1.05μ)
    t = @elapsed for _ in 1:samples
        primitive_from_conserved(eos, N, Mx, My, E; T0=1.01T, μ0=1.05μ)
    end
    println(name)
    println("  cold start : ", cold, "  in ", itc, " iterations")
    println("  warm start : ", warm, "  in ", itw, " iterations")
    println("  max |Δ|    : ", maximum(abs.(warm .- SVector(T, μ, vx, vy))))
    println("  warm cost  : ", round(1e9*t/samples, digits=1), " ns, ", allocs, " B allocated")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo("IdealQCD (baryon charge)", IdealQCD(), 0.25, 0.08, 0.3, -0.2)
    demo("HeavyQuark (charm charge)", HeavyQuark(), 0.25, 0.08, 0.3, -0.2)
end
