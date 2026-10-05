struct ExampleSecondOrder <: AbstractSecondOrderTransport end

function FluidumThermodynamics.second_order_state(::ExampleSecondOrder, x, th, leading)
    SecondOrderState(:custom_relaxation;
        shear=(delta_pipi=2*leading.shear.tau,),
        bulk=(lambda_Pipi=leading.bulk.tau,),
    )
end

struct CountedTransportEOS <: AbstractEquationOfState{NaturalGeV}
    calls::Base.RefValue{Int}
    unit_scale::Float64
end
CountedTransportEOS(calls) = CountedTransportEOS(calls,unit_scale(NaturalGeV()))

function FluidumThermodynamics.thermodynamic(eos::CountedTransportEOS, x::GrandCanonicalPoint)
    eos.calls[] += 1
    thermodynamic(IdealQCD(),x)
end

struct CountedShear <: AbstractShearTransport
    calls::Base.RefValue{Int}
end

function FluidumThermodynamics.shear_state(model::CountedShear, x, th)
    model.calls[] += 1
    FluidumThermodynamics.shear_state(ConstantEtaOverS(0.16,5.0),x,th)
end

@testset "Optional second-order transport" begin
    x = GrandCanonicalPoint(0.2,0.03)
    eos = IdealQCD()
    shear, bulk, charge = ConstantEtaOverS(0.16,5.0), ConstantZetaOverS(0.03,10.0), ZeroChargeTransport()
    old_model = TransportModel(shear,bulk,charge)
    th, old = @inferred transport(FluidModel(eos,old_model),x)
    @test old_model.second_order isa NoSecondOrderTransport
    @test old.second_order === nothing
    @test TransportState(old.shear,old.bulk,old.charge).second_order === nothing

    custom_model = TransportModel(shear,bulk,charge; second_order=ExampleSecondOrder())
    _, tr = @inferred transport(FluidModel(eos,custom_model),x)
    @test tr.shear == old.shear
    @test tr.bulk == old.bulk
    @test tr.charge == old.charge
    @test tr.second_order.basis === :custom_relaxation
    @test tr.second_order.shear.delta_pipi == 2*old.shear.tau
    @test tr.second_order.bulk.lambda_Pipi == old.bulk.tau
    @test isempty(tr.second_order.charge)
    @test !hasproperty(tr.second_order.shear,:lambda_piPi)

    # Coefficients are model data: the structure does not impose sign constraints.
    prescribed = ConstantSecondOrderTransport(:example;
        shear=(lambda_piPi=-0.05,), bulk=(lambda_Pipi=0.03,))
    _, tr = @inferred transport(FluidModel(eos,TransportModel(shear,bulk,charge,prescribed)),x)
    @test tr.second_order === prescribed.state
    @test tr.second_order.shear.lambda_piPi == -0.05
    @test isempty(SecondOrderState(:empty).shear)
end

@testset "Multi-charge second-order couplings" begin
    x = GrandCanonicalPoint(0.2,0.03,-0.01)
    eos = PolynomialMultiChargeEOS(3.0,@SMatrix [2.0 0.3; 0.3 1.5])
    matrix = @SMatrix [0.2 0.03; -0.01 0.4]
    vector = @SVector [0.1,0.2]
    tensor = reshape(collect(1.0:8.0),2,2,2)
    state = SecondOrderState(:multicharge_example;
        shear=(lambda_piPi=0.1, ell_piV=vector, phi_VV=matrix),
        bulk=(lambda_Pipi=0.2, ell_PiV=vector),
        charge=(delta_VV=matrix, ell_VPi=vector, nonlinear_VVV=tensor),
    )
    second = ConstantSecondOrderTransport(state)
    tm = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),second)
    _, tr = @inferred transport(FluidModel(eos,tm),x)
    @test tr.second_order.shear.phi_VV == matrix
    @test tr.second_order.bulk.ell_PiV == vector
    @test tr.second_order.charge.delta_VV[1,2] == matrix[1,2]
    @test tr.second_order.charge.nonlinear_VVV === tensor

    # Validate every charge axis, including arrays in shear/bulk equations.
    bad_states = (
        SecondOrderState(:bad; charge=(delta_VV=ones(2,3),)),
        SecondOrderState(:bad; charge=(nonlinear_VVV=ones(2,2,3),)),
        SecondOrderState(:bad; shear=(ell_piV=ones(3),)),
        SecondOrderState(:bad; bulk=(phi_VV=ones(3,3),)),
    )
    for bad in bad_states
        tm_bad = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),
                                ConstantSecondOrderTransport(bad))
        @test_throws DimensionMismatch transport(FluidModel(eos,tm_bad),x)
    end
    @test_throws ArgumentError SecondOrderState(:bad; charge=(delta_VV=0.1,))
    @test_throws ArgumentError SecondOrderState(:bad; shear=(delta_pipi="0.1",))
    @test_throws ArgumentError SecondOrderState(:bad; bulk=(ell_PiV=["a","b"],))
    @test_throws ArgumentError SecondOrderState(:bad; charge=(value=fill(1.0),))

    # A single charge still has explicit charge axes.
    onecharge = ConstantSecondOrderTransport(:one;
        charge=(delta_VV=SMatrix{1,1}(0.2),ell_VPi=SVector(0.1)))
    tm_one = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),onecharge)
    _, tr_one = transport(FluidModel(IdealQCD(),tm_one),GrandCanonicalPoint(0.2,0.03))
    @test size(tr_one.second_order.charge.delta_VV) == (1,1)

    # No-charge models can omit the sector or carry correctly typed empty arrays.
    zerocharge = ConstantSecondOrderTransport(:zero;
        charge=(delta_VV=SMatrix{0,0,Float64}(),))
    tm_zero = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),zerocharge)
    _, tr_zero = transport(FluidModel(IdealQCD(),tm_zero),GrandCanonicalPoint(0.2))
    @test size(tr_zero.second_order.charge.delta_VV) == (0,0)
    @test_throws DimensionMismatch transport(FluidModel(IdealQCD(),tm_one),GrandCanonicalPoint(0.2))
end

@testset "State-dependent evaluation reuses EOS and transport" begin
    eos_calls, shear_calls, callback_calls = Ref(0),Ref(0),Ref(0)
    eos = CountedTransportEOS(eos_calls)
    second = FunctionalSecondOrderTransport() do x,th,leading
        callback_calls[] += 1
        @test leading.second_order === nothing
        SecondOrderState(:state_dependent;
            shear=(delta_pipi=x.T*leading.shear.tau,),
            bulk=(example=th.pressure,),
            charge=(delta_VV=susceptibilities(th),),
        )
    end
    tm = TransportModel(CountedShear(shear_calls),ZeroBulkTransport(),ZeroChargeTransport(),second)
    x = GrandCanonicalPoint(0.2,0.03)
    th,tr = @inferred transport(FluidModel(eos,tm),x)
    @test (eos_calls[],shear_calls[],callback_calls[]) == (1,1,1)
    @test tr.second_order.shear.delta_pipi == x.T*tr.shear.tau
    @test tr.second_order.bulk.example == th.pressure
    @test tr.second_order.charge.delta_VV == susceptibilities(th)
    transport(tm,x,th)
    @test (eos_calls[],shear_calls[],callback_calls[]) == (1,2,2)

    bad = FunctionalSecondOrderTransport((x,th,leading) -> (coefficient=0.1,))
    tm_bad = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),bad)
    @test_throws ArgumentError transport(tm_bad,x,th)
    wrong_dimensions = FunctionalSecondOrderTransport((x,th,leading) ->
        SecondOrderState(:bad; charge=(delta_VV=ones(2,2),)))
    tm_bad_dims = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),wrong_dimensions)
    @test_throws DimensionMismatch transport(tm_bad_dims,x,th)

    # Numeric types survive the callback; model coefficients can be differentiated.
    differentiable = FunctionalSecondOrderTransport((x,th,leading) ->
        SecondOrderState(:analytic_example; shear=(example=x.T^2,)))
    tm_ad = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport(),differentiable)
    fluid_ad = FluidModel(IdealQCD(),tm_ad)
    coefficient(t) = transport(fluid_ad,GrandCanonicalPoint(t))[2].second_order.shear.example
    @test ForwardDiff.derivative(coefficient,0.2) ≈ 0.4
    @test coefficient(big"0.2") isa BigFloat
end
