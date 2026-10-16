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

@inline function _check_heavy_quark_charges(M)
    M <= 1 || throw(DimensionMismatch("HeavyQuark carries a single charge; the point has $M chemical potentials"))
end

# The quark gas takes the point as is: an omitted chemical potential is zero.
@inline function pressure(eos::HeavyQuark, x::GrandCanonicalPoint{R,M}) where {R,M}
    _check_heavy_quark_charges(M)
    p = pressure(eos.bulk,GrandCanonicalPoint(x.T))
    backreaction(eos) ? p + pressure(eos.charm,x) : p
end

function thermodynamic(eos::HeavyQuark, x::GrandCanonicalPoint{R,M}) where {R,M}
    _check_heavy_quark_charges(M)
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

"""
    CharmTransport(mass, DsT)
    CharmTransport(hq::HeavyQuark, DsT)

Free (Boltzmann) heavy-quark diffusion, the prescription of Fluidum. It is
`ConstantDsT` with the relaxation time set by the kinetic-theory Bessel
formula instead of a fixed front speed: at each state

    Ctau = 48 K₂(z) / (z³ (2K₁(z) - 3K₃(z) + K₅(z))),   z = mass/T

For a Boltzmann gas, where `T·∂n/∂μ = n`, this gives Fluidum's
`τ = DₛT·z³/(48T)·(2K₁ - 3K₃ + K₅)/K₂`, which tends to `Dₛ·mass/T` for heavy
quarks. As for `ConstantDsT`, `κ = DₛT·n/T` from the state's densities
applies to every charge of the fluid, in the unit convention of the state;
the model is meant for a fluid whose charge is the heavy quark, such as
`HeavyQuark`, from which the mass can be taken directly. `DsT` is the
dimensionless `DₛT`, a number or a callable `T -> DₛT`; `mass` is in the EOS
energy unit.
"""
struct CharmTransport{M,D} <: AbstractChargeTransport
    mass::M
    DsT::D   # callable T -> DₛT
    function CharmTransport(mass::Real, DsT)
        isfinite(mass) && mass > 0 || throw(ArgumentError("heavy-quark mass must be finite and positive"))
        new{typeof(mass),typeof(DsT)}(mass,DsT)
    end
end

CharmTransport(mass::Real, DsT::Real) = CharmTransport(mass,Returns(DsT))
CharmTransport(hq::HeavyQuark, DsT) = CharmTransport(heavy_quark_mass(hq),DsT)

@inline function charge_state(model::CharmTransport, x::GrandCanonicalPoint, th)
    z = model.mass/x.T
    # Scaled Bessel functions: the common e^z factor cancels in the ratio.
    b1, b2 = besselkx(1,z), besselkx(2,z)
    b3 = muladd(4/z,b2,b1)
    b4 = muladd(6/z,b3,b2)
    b5 = muladd(8/z,b4,b3)
    Ctau = 48*b2/(z^3*(2*b1-3*b3+b5))
    charge_state(ConstantDsT(model.DsT(x.T),Ctau),x,th)
end
