using FluidumThermodynamics
using StaticArrays

χ = @SMatrix [2.0 0.3; 0.3 1.5]
eos = PolynomialMultiChargeEOS(3.0,χ)
x = GrandCanonicalPoint(0.2,0.03,-0.01)
th = thermodynamic(eos,x)
println("Pressure: ",th.pressure)
println("Entropy density: ",entropy_density(th))
println("Charge densities: ",charge_densities(th))
println("Energy density: ",energy_density(x,th))

target = microcanonical(x,th)
recovered = grandcanonical(eos,target,GrandCanonicalPoint(0.18,0.0,0.0))
println("Recovered temperature: ",recovered.T)
println("Recovered chemical potentials: ",recovered.μ)

model = TransportModel(
    ConstantEtaOverS(0.16,5.0),
    ConstantZetaOverS(0.03,10.0),
    ConstantChargeTransport(
        @SMatrix([0.4 0.05; 0.05 0.3]),
        @SMatrix([1.0 0.0; 0.0 1.2]),
    ),
)
_,tr = transport(FluidModel(eos,model),x)
println("Shear viscosity: ",tr.shear.eta)
println("Conductivity matrix: ",tr.charge.kappa)
