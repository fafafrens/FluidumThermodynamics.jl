"""
    HeavyQuark(; mass=1.5, bulk=FluidumEOS(), units=unit_system(bulk), backreaction=true)
    HeavyQuark(bulk, charm; backreaction=true)

Bulk EOS carrying one conserved heavy-quark charge, modeled on Fluidum's
`Heavy_Quark`. `bulk` is evaluated at zero chemical potential; `charm` is a
Boltzmann gas of a single heavy-quark species with degeneracy 6 (spin × color)
and unit charge at chemical potential `μ[1]`, and supplies the charge sector
(density, its temperature derivative, and susceptibility). Antiquarks are not
included.

With `backreaction=true` the pressure is `p_bulk(T) + p_charm(T, μ)`, so the
state derives from a single potential and the Maxwell relations and
Gibbs-Duhem hold; the quark gas then also contributes to entropy and heat
capacity. With `backreaction=false` the temperature sector is the bulk alone,
as in Fluidum, where the quark gas does not back-react on the bulk.

Points may omit the chemical potential, which is then zero and the state
carries temperature derivatives only. `mass` uses the energy unit of `units`;
`bulk` is converted to `units` when its convention differs.
"""
struct HeavyQuark{U,B<:AbstractEquationOfState{U},C<:HadronResonanceGas{U},S,R} <: AbstractEquationOfState{U}
    bulk::B
    charm::C
    unit_scale::S
    function HeavyQuark(bulk::AbstractEquationOfState{U}, charm::HadronResonanceGas{V};
                        backreaction::Bool=true) where {U,V}
        U === V || throw(ArgumentError(
            "bulk and heavy-quark gas use different unit systems ($U and $V); use convert_units first"))
        length(charm.particles) == 1 || throw(ArgumentError("heavy-quark gas must contain exactly one species"))
        length(charm.particles[1].charges) == 1 || throw(DimensionMismatch("heavy-quark gas must carry a single charge"))
        scale = unit_scale(bulk)
        new{U,typeof(bulk),typeof(charm),typeof(scale),backreaction}(bulk,charm,scale)
    end
end

function HeavyQuark(; mass=1.5, bulk::AbstractEquationOfState=FluidumEOS(),
                    units::AbstractUnitSystem=unit_system(bulk), backreaction::Bool=true)
    charm = HadronResonanceGas([HadronSpecies(mass,6,(1,); statistics=:boltzmann)]; units)
    HeavyQuark(convert_units(units,bulk),charm; backreaction)
end

"""Whether the heavy-quark gas contributes to the pressure and its temperature derivatives."""
@inline backreaction(::HeavyQuark{U,B,C,S,R}) where {U,B,C,S,R} = R

# Charm mass in the EOS energy unit, used by `CharmTransport`.
@inline heavy_quark_mass(eos::HeavyQuark) = eos.charm.particles[1].mass

# The quark gas takes the point as is: an omitted chemical potential is zero,
# and it rejects points with more than one charge.
@inline function pressure(eos::HeavyQuark, x::GrandCanonicalPoint)
    p = pressure(eos.bulk,GrandCanonicalPoint(x.T))
    backreaction(eos) ? p + pressure(eos.charm,x) : p
end

function thermodynamic(eos::HeavyQuark, x::GrandCanonicalPoint{R,M}) where {R,M}
    bulk = thermodynamic(eos.bulk,GrandCanonicalPoint(x.T))
    charm = thermodynamic(eos.charm,x)
    if backreaction(eos)
        p = bulk.pressure + charm.pressure
        s = bulk.gradient[1] + charm.gradient[1]
        tt = bulk.hessian[1,1] + charm.hessian[1,1]
    else
        p, s, tt = bulk.pressure, bulk.gradient[1], bulk.hessian[1,1]
    end
    M == 0 && return ThermodynamicState(eos,p,SVector(s),SMatrix{1,1}(tt))
    # Charge sector from the quark gas. The mixed entry ∂n/∂T = ∂s/∂μ keeps the
    # Hessian symmetric; without back-reaction it is not the Hessian of `p`.
    g = SVector(s,charm.gradient[2])
    H = SMatrix{2,2}(tt,charm.hessian[2,1],charm.hessian[1,2],charm.hessian[2,2])
    ThermodynamicState(eos,p,g,H)
end

# Same composition as `thermodynamic`, one order higher: the TTT entry comes from the
# bulk fluid (plus the quark gas with back-reaction); every entry with a charge index
# comes from the quark gas alone.
function thermodynamic_perturbation(eos::HeavyQuark, x::GrandCanonicalPoint{R,M}) where {R,M}
    bulk = thermodynamic_perturbation(eos.bulk,GrandCanonicalPoint(x.T))
    charm = thermodynamic_perturbation(eos.charm,x)
    if backreaction(eos)
        p = bulk.pressure + charm.pressure
        s = bulk.gradient[1] + charm.gradient[1]
        tt = bulk.hessian[1,1] + charm.hessian[1,1]
        ttt = bulk.third[1,1,1] + charm.third[1,1,1]
    else
        p, s, tt, ttt = bulk.pressure, bulk.gradient[1], bulk.hessian[1,1], bulk.third[1,1,1]
    end
    M == 0 && return ThermodynamicPerturbationState(eos,p,SVector(s),SMatrix{1,1}(tt),SArray{Tuple{1,1,1}}(ttt))
    g = SVector(s,charm.gradient[2])
    H = SMatrix{2,2}(tt,charm.hessian[2,1],charm.hessian[1,2],charm.hessian[2,2])
    # third[i,j,l] = ∂_l hessian[i,j] of the composed state: the TT entry is bulk (+ quark gas
    # with back-reaction), every entry with a charge index is the quark gas. Without
    # back-reaction the tensor is symmetric in (i,j) only, as the Hessian is then not the
    # Hessian of the pressure.
    C = SArray{Tuple{2,2,2}}(ntuple(Val(8)) do k
        i, j, l = Tuple(CartesianIndices((2,2,2))[k])
        if i == 1 && j == 1
            l == 1 ? ttt : (backreaction(eos) ? charm.third[1,1,2] : zero(ttt))
        else
            charm.third[i,j,l]
        end
    end)
    ThermodynamicPerturbationState(eos,p,g,H,C)
end

"""
    CharmCtau(mass)

Relaxation-time parameter `Ctau(x, th)` of free (Boltzmann) heavy-quark
diffusion for `EinsteinDiffusion`, Fluidum's kinetic-theory formula

    Ctau = 48 K₂(z) / (z³ (2K₁(z) - 3K₃(z) + K₅(z))),   z = mass/T

For a Boltzmann gas, where `T·∂n/∂μ = n`, this gives
`τ = DₛT·z³/(48T)·(2K₁ - 3K₃ + K₅)/K₂`, which tends to `Dₛ·mass/T` for heavy
quarks. `mass` is in the EOS energy unit.
"""
struct CharmCtau{M}
    mass::M
    function CharmCtau(mass::Real)
        isfinite(mass) && mass > 0 || throw(ArgumentError("heavy-quark mass must be finite and positive"))
        new{typeof(mass)}(mass)
    end
end

@inline function (c::CharmCtau)(x, th)
    z = c.mass/x.T
    # Scaled Bessel functions: the common e^z factor cancels in the ratio.
    b1, b2 = besselkx(1,z), besselkx(2,z)
    b3 = muladd(4/z,b2,b1)
    b4 = muladd(6/z,b3,b2)
    b5 = muladd(8/z,b4,b3)
    48*b2/(z^3*(2*b1-3*b3+b5))
end

"""
    CharmTransport(mass, DsT)
    CharmTransport(hq::HeavyQuark, DsT)

Free (Boltzmann) heavy-quark diffusion, the prescription of Fluidum:
`EinsteinDiffusion(DsT, CharmCtau(mass))`, that is `κ = DₛT·n/T` from the
state's densities for every charge of the fluid, with the relaxation time of
`CharmCtau`, in the unit convention of the state. The model is meant for a
fluid whose charge is the heavy quark, such as `HeavyQuark`, from which the
mass can be taken directly. `DsT` is the dimensionless `DₛT`, a number or a
callable `(x, th)` as in `EinsteinDiffusion`.
"""
CharmTransport(mass::Real, DsT) = EinsteinDiffusion(DsT,CharmCtau(mass))
CharmTransport(hq::HeavyQuark, DsT) = CharmTransport(heavy_quark_mass(hq),DsT)
