using FluidumThermodynamics
using StaticArrays

# A phenomenological example; the ratios below are demonstration inputs,
# not a kinetic-theory closure or a calibrated parameter set.
ratios = (delta_pipi=1.1, tau_pipi=1.3, lambda_piPi=0.2,
          delta_PiPi=0.9, lambda_Pipi=0.1, delta_VV=1.2)

second = FunctionalSecondOrderTransport() do x,th,leading
    SecondOrderState(:example_landau_relaxation;
        shear=(
            delta_pipi=ratios.delta_pipi*leading.shear.tau,
            tau_pipi=ratios.tau_pipi*leading.shear.tau,
            lambda_piPi=ratios.lambda_piPi*leading.shear.tau,
        ),
        bulk=(
            delta_PiPi=ratios.delta_PiPi*leading.bulk.tau,
            lambda_Pipi=ratios.lambda_Pipi*leading.bulk.tau,
        ),
        charge=(delta_VV=ratios.delta_VV*leading.charge.tau,),
    )
end

eos = PolynomialMultiChargeEOS(3.0,@SMatrix [2.0 0.3; 0.3 1.5])
model = TransportModel(
    ConstantEtaOverS(0.16,5.0),
    ConstantZetaOverS(0.03,10.0),
    ConstantChargeTransport(
        @SMatrix([0.4 0.05; 0.05 0.3]),
        @SMatrix([1.0 0.02; 0.02 1.2]),
    );
    second_order=second,
)

x = GrandCanonicalPoint(0.2,0.03,-0.01)
th,tr = transport(FluidModel(eos,model),x)
println("Convention: ",tr.second_order.basis)
println("Shear coefficients: ",tr.second_order.shear)
println("Bulk coefficients: ",tr.second_order.bulk)
println("Diffusion-expansion matrix: ",tr.second_order.charge.delta_VV)

# Contract one of the stored charge matrices with a sample diffusion-current
# component and expansion rate. Spacetime projections belong to the caller.
V = @SVector [0.01,-0.02]
θ = 0.1
diffusion_expansion = -θ*(tr.second_order.charge.delta_VV*V)
println("Diffusion-expansion contribution: ",diffusion_expansion)
