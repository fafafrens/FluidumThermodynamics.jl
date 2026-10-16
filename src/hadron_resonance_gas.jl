"""
    HadronSpecies(mass, degeneracy, charges=(); statistics=:boltzmann)

A zero-width resonance. Mass uses the EOS energy unit; charges follow the same
basis as the point's chemical potentials. Include antiparticles explicitly.
Statistics may be `:boltzmann`, `:boson`, or `:fermion`.
"""
struct HadronSpecies{T,N}
    mass::T
    degeneracy::T
    charges::SVector{N,T}
    statistics::Int8
end

function HadronSpecies(mass, degeneracy, charges=(); statistics=:boltzmann)
    sign = statistics === :boltzmann ? 0 : statistics === :boson ? 1 :
           statistics === :fermion ? -1 : throw(ArgumentError("unknown statistics: $statistics"))
    m,g = promote(float(mass),float(degeneracy))
    isfinite(m) && m > 0 || throw(ArgumentError("mass must be finite and positive"))
    isfinite(g) && g > 0 || throw(ArgumentError("degeneracy must be finite and positive"))
    q = SVector{length(charges),typeof(m)}(charges)
    HadronSpecies(m,g,q,Int8(sign))
end

"""
    HadronResonanceGas(particles; units=NaturalGeV(), terms=16)

Ideal gas of the supplied resonances, with analytic pressure gradient and Hessian.
`terms` truncates the quantum fugacity expansion; Boltzmann species use one term.
Masses, temperature, and chemical potentials use the selected energy unit.
Quantum species require `dot(charges, μ) < mass`; increase `terms` to check
convergence near that boundary. No spectrum or antiparticles are added implicitly.
Points may supply the first `M` chemical potentials in the charge basis; omitted
chemical potentials are zero. The state contains derivatives for `(T, μ₁, …, μₘ)`.
"""
struct HadronResonanceGas{U,T,N,S} <: AbstractEquationOfState{U}
    particles::Vector{HadronSpecies{T,N}}
    terms::Int
    unit_scale::S
end

function HadronResonanceGas(particles; units::U=NaturalGeV(), terms::Integer=16) where {U<:AbstractUnitSystem}
    ps = collect(particles)
    isempty(ps) && throw(ArgumentError("supply at least one hadron species"))
    terms > 0 || throw(ArgumentError("terms must be positive"))
    N = length(first(ps).charges)
    all(p -> length(p.charges) == N, ps) || throw(DimensionMismatch("species must share a charge basis"))
    T = promote_type(map(p -> typeof(p.mass),ps)...)
    typed = [HadronSpecies(T(p.mass),T(p.degeneracy),SVector{N,T}(p.charges),p.statistics) for p in ps]
    scale = unit_scale(units)
    HadronResonanceGas{U,T,N,typeof(scale)}(typed,Int(terms),scale)
end

function thermodynamic(eos::HadronResonanceGas{U,E,N},
                       x::GrandCanonicalPoint{R,M}) where {U,E,N,R,M}
    M <= N || throw(DimensionMismatch("point has more charges than the EOS basis"))
    T = x.T
    T > 0 || throw(DomainError(T,"HRG requires positive temperature"))
    V = promote_type(E,R,typeof(eos.unit_scale))
    p = zero(V)
    g = zero(SVector{M+1,V})
    H = zero(SMatrix{M+1,M+1,V})
    invT = inv(T)
    for species in eos.particles
        q = SVector{M,E}(ntuple(i -> species.charges[i], M))
        μ = dot(q,x.μ)
        m = species.mass
        if species.statistics != 0 && μ >= m
            throw(DomainError(μ,"quantum fugacity expansion requires species chemical potential < mass"))
        end
        count = species.statistics == 0 ? 1 : eos.terms
        prefactor = eos.unit_scale*species.degeneracy*m^2/(2*pi^2)
        for k in 1:count
            z = k*m*invT
            a = k*μ*invT
            sign = species.statistics == -1 && iseven(k) ? -1 : 1
            c = sign*prefactor*exp(k*(μ-m)*invT)/(k*k)
            # Scaled Bessel functions avoid underflow times overflowing fugacity.
            b1 = besselkx(1,z)
            b2 = besselkx(0,z)+2*b1/z
            p += c*T^2*b2
            s = c*T*((4-a)*b2+z*b1)
            tt = c*((12-6*a+a*a+z*z)*b2+(3-2*a)*z*b1)
            mixed = c*k*((3-a)*b2+z*b1)
            density = c*T*k*b2
            chi = c*k*k*b2
            g += SVector{M+1}(ntuple(i -> i == 1 ? s : density*q[i-1], Val(M+1)))
            H += SMatrix{M+1,M+1}(ntuple(Val((M+1)^2)) do index
                i = mod1(index,M+1)
                j = (index-1) ÷ (M+1)+1
                i == 1 ? (j == 1 ? tt : mixed*q[j-1]) :
                j == 1 ? mixed*q[i-1] : chi*q[i-1]*q[j-1]
            end)
        end
    end
    ThermodynamicState(eos,p,g,H)
end

pressure(eos::HadronResonanceGas,x::GrandCanonicalPoint) = thermodynamic(eos,x).pressure

"""
    HeavyQuark(; mass=1.5, units=NaturalGeV())

Boltzmann gas of a single heavy-quark species with degeneracy 6 (spin × color)
and unit charge, implemented as a one-species `HadronResonanceGas`. The mass
uses the selected energy unit. Antiquarks are not included.
"""
HeavyQuark(; mass=1.5, units::AbstractUnitSystem=NaturalGeV()) =
    HadronResonanceGas([HadronSpecies(mass,6,(1,); statistics=:boltzmann)]; units)

# Charm mass of a single-species heavy-quark EOS, used by `CharmTransport`.
function heavy_quark_mass(eos::HadronResonanceGas)
    length(eos.particles) == 1 || throw(ArgumentError("heavy-quark EOS must contain exactly one species"))
    eos.particles[1].mass
end
