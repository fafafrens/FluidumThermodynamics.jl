"""
    AbstractUnitSystem

Type-level convention for plain numerical EOS inputs and thermodynamic
outputs. Unit systems are zero-field singleton types. Temperature is expressed
in energy units (`k_B = 1`), conserved charges are dimensionless, and pressure
has units of energy per volume.

Custom conventions implement `energy_scale(units)` (one energy unit in GeV)
and `length_scale(units)` (one length unit in fm), returning positive scales;
`hbarc(units)` and `unit_scale(units)` follow from them.
"""
abstract type AbstractUnitSystem end

"""Natural units: GeV for energy, GeV⁻¹ for length, GeV⁴ for pressure (ℏ = c = k_B = 1)."""
struct NaturalGeV <: AbstractUnitSystem end

"""GeV for temperature/chemical potentials, fm for length, GeV/fm³ for pressure."""
struct GeVFm <: AbstractUnitSystem end

"""MeV for temperature/chemical potentials, fm for length, MeV/fm³ for pressure."""
struct MeVFm <: AbstractUnitSystem end

"""Size of a convention's energy unit in GeV."""
function energy_scale end
energy_scale(::GeVFm) = 1//1
energy_scale(::NaturalGeV) = 1//1
energy_scale(::MeVFm) = 1//1000

"""Size of a convention's length unit in fm."""
function length_scale end
length_scale(::GeVFm) = 1//1
length_scale(::NaturalGeV) = _hbarc_GeV_fm
length_scale(::MeVFm) = 1//1

# Preserve the original Fluidum value of ℏc (GeV fm).
const _hbarc_GeV_fm = 0.1973261

"""
    hbarc(units::AbstractUnitSystem)

ℏc in the convention's units, energy unit × length unit: 1 for `NaturalGeV`,
0.1973 GeV fm for `GeVFm`, 197.3 MeV fm for `MeVFm`. A quantity computed with
ℏ = c = 1 in the energy unit that carries lengthᵇ is multiplied by
`hbarc(units)^b` to land in the convention: pressure (length⁻³) by
`hbarc^-3`, a conductivity (length⁻²) by `hbarc^-2`, a relaxation time
(length) by `hbarc`.
"""
@inline hbarc(units::AbstractUnitSystem) = _hbarc_GeV_fm/(energy_scale(units)*length_scale(units))
@inline hbarc(::NaturalGeV) = 1.0

"""
    unit_scale(units::AbstractUnitSystem)
    unit_scale(eos::AbstractEquationOfState)

Conversion factor from pressure in energy⁴ to energy/length³, `hbarc(units)^-3`.
EOS constructors store this factor in their required `unit_scale` field;
`unit_scale(eos)` reads the stored value.
"""
@inline unit_scale(units::AbstractUnitSystem) = hbarc(units)^-3
@inline unit_scale(::NaturalGeV) = 1.0

"""
    unit_system(eos)
    unit_system(th::AbstractThermodynamicState)

Return the singleton unit convention encoded in the object's type.
"""
function unit_system end

"""
    convert_units(target, eos)
    convert_units(target, th::AbstractThermodynamicState)
    convert_units(target, source, point)

Return an EOS, evaluated state, or coordinate point in the target convention,
rescaling numerical values rather than changing only a label. Coordinate
points contain plain numbers without a unit tag, so their source convention
must be supplied explicitly. Inputs to an EOS always use its convention.
"""
function convert_units end

@inline function _conversion_factors(source::AbstractUnitSystem, target::AbstractUnitSystem)
    es,ls = energy_scale(source),length_scale(source)
    et,lt = energy_scale(target),length_scale(target)
    energy = es/et
    density = (lt/ls)^3
    # Each derivative with respect to a grand-canonical coordinate divides by one energy unit.
    (energy=energy, pressure=energy*density, density=density,
     hessian=density/energy, third=density/energy^2)
end
