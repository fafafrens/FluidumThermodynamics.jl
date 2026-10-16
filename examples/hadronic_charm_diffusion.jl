# Hadronic charm diffusion with a state-dependent relaxation time.
#
# A callable struct closes over the charm hadron list and feeds
# `EinsteinDiffusion` as its `Ctau`. This reproduces Fluidum's
# `τ_diffusion_hadron`: κ = DₛT·n/T from the fluid's charm density and
# τ = Σ qᵢ²nᵢτᵢ / Σ qᵢ²nᵢ, the χ-weighted mean of the single-species
# relaxation times τᵢ = DₛT zᵢ³/(48T) (2K₁-3K₃+K₅)/K₂, zᵢ = mᵢ/T.
#
# Run with `julia --project=. examples/hadronic_charm_diffusion.jl`; including
# the file only defines `CharmHadronCtau`.

using FluidumThermodynamics

"""
    CharmHadronCtau(species)

`Ctau(x, th)` for `EinsteinDiffusion` built from a list of charm
`HadronSpecies` with one charge, the heavy-quark count `q` of each hadron.
For a Boltzmann gas `T·∂n/∂μ = Σq²n`, so the χ-weighted mean relaxation time
`τ = Σq²nᵢτᵢ/Σq²nᵢ` corresponds to `Ctau = κ/(τ·T·∂n/∂μ) = Σqn/(T·Σq²nᵢτ̂ᵢ)`,
with `τ̂ᵢ` the single-species factor per unit `DₛT`. The unit scale of the
densities cancels in the ratio.
"""
struct CharmHadronCtau{T}
    species::Vector{HadronSpecies{T,1}}
end

function (c::CharmHadronCtau)(x, th)
    T = x.T
    α = x.μ[1]/T
    density = zero(α)
    weighted = zero(α)
    for s in c.species
        q = s.charges[1]
        z = s.mass/T
        b1, b2 = FluidumThermodynamics.besselkx(1,z), FluidumThermodynamics.besselkx(2,z)
        b3 = b1 + 4/z*b2
        b4 = b2 + 6/z*b3
        b5 = b3 + 8/z*b4
        n = s.degeneracy*s.mass^2*T/(2*pi^2)*exp(q*α - z)*b2
        τ̂ = z^3/(48*T)*(2*b1 - 3*b3 + b5)/b2
        density += q*n
        weighted += q^2*n*τ̂
    end
    density/(T*weighted)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Open-charm ground states and the J/ψ (q = 2), masses in GeV.
    hadrons = [
        HadronSpecies(1.870,2,(1,)),   # D⁺, D⁰ (one charge state each: antiparticles carry the same q)
        HadronSpecies(2.010,6,(1,)),   # D*
        HadronSpecies(1.968,1,(1,)),   # Dₛ
        HadronSpecies(2.286,2,(1,)),   # Λc
        HadronSpecies(3.097,3,(2,)),   # J/ψ
    ]
    eos = HadronResonanceGas(hadrons)
    DsT = 0.2
    model = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),
                           EinsteinDiffusion(DsT,CharmHadronCtau(hadrons)))
    fluid = FluidModel(eos,model)
    # The free quark for comparison, evaluated on its own single-species gas.
    quark_species = [HadronSpecies(1.5,6,(1,))]
    quark = FluidModel(HadronResonanceGas(quark_species),
                       TransportModel(ZeroShearTransport(),ZeroBulkTransport(),
                                      EinsteinDiffusion(DsT,CharmHadronCtau(quark_species))))
    println("T [GeV]   n_c [GeV³]     κ [GeV²]     τ_hadrons [GeV⁻¹]   τ_free quark [GeV⁻¹]")
    for T in (0.12,0.14,0.16,0.18)
        x = GrandCanonicalPoint(T,0.0)
        th, tr = transport(fluid,x)
        _, trq = transport(quark,x)
        println(rpad(T,9)," ",rpad(round(charge_density(th,1),sigdigits=4),14),
                rpad(round(tr.charge.kappa[1,1],sigdigits=4),13),
                rpad(round(tr.charge.tau[1,1],sigdigits=4),20),
                round(trq.charge.tau[1,1],sigdigits=4))
    end
end
