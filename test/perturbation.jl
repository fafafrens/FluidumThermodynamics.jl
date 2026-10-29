const FT = FluidumThermodynamics

# Central finite difference of `f(y)` along coordinate `k`.
function fd_coordinate(f, y::SVector{D}, k; h=1e-5) where {D}
    e = SVector{D}(ntuple(i -> i == k ? one(eltype(y)) : zero(eltype(y)), D))
    (f(y + h*e) - f(y - h*e))/(2h)
end

hessian_at(eos) = z -> thermodynamic(eos, FT.grandcanonical_from_coordinates(z)).hessian

@testset "Third derivatives: polynomial EOS, exact" begin
    for N in 0:2
        χ = SMatrix{N,N,Float64}(Matrix{Float64}(I,N,N))
        eos = PolynomialMultiChargeEOS(3.0, χ)
        μ = SVector{N,Float64}(ntuple(i -> 0.01*i, N))
        x = GrandCanonicalPoint(0.2, μ)
        th = thermodynamic(eos,x)
        th3 = @inferred thermodynamic_perturbation(eos,x)
        @test th3 isa ThermodynamicPerturbationState
        @test th3 isa AbstractThermodynamicState
        @test th3.pressure ≈ th.pressure
        @test th3.gradient ≈ th.gradient
        @test th3.hessian ≈ th.hessian
        # p = 3T⁴ + T² μᵀχμ/2
        @test third_derivative(th3,1,1,1) ≈ 72*x.T
        for i in 1:N
            @test third_derivative(th3,1,1,i+1) ≈ 2*(χ*μ)[i]
            for j in 1:N
                @test third_derivative(th3,1,i+1,j+1) ≈ 2*x.T*χ[i,j]
                for k in 1:N
                    @test abs(third_derivative(th3,i+1,j+1,k+1)) < 1e-12
                end
            end
        end
        @test third_derivatives(th3) === th3.third
        @test size(th3.third) == (N+1,N+1,N+1)
        # accessors shared with ThermodynamicState
        @test entropy_density(th3) == entropy_density(th)
        @test charge_densities(th3) == charge_densities(th)
        @test energy_density(x,th3) ≈ energy_density(x,th)
        @test speed_of_sound_squared(x,th3) ≈ speed_of_sound_squared(x,th)
        @test unit_system(th3) === unit_system(th)
        projected = ThermodynamicState(th3)
        @test projected isa ThermodynamicState
        @test projected.hessian == th3.hessian
    end
end

@testset "Third derivatives against finite differences of the Hessian" begin
    hrg = HadronResonanceGas([HadronSpecies(0.14,1,(1,0)),HadronSpecies(0.49,2,(-1,1))]; terms=16)
    cases = ((IdealQCD(), GrandCanonicalPoint(0.2)),
             (IdealQCD(), GrandCanonicalPoint(0.2,0.05)),
             (FluiduMEoS(), GrandCanonicalPoint(0.25)),
             (hrg, GrandCanonicalPoint(0.15,0.02,-0.01)),
             (HeavyQuark(), GrandCanonicalPoint(0.25)),
             (HeavyQuark(), GrandCanonicalPoint(0.25,0.1)),
             (HeavyQuark(backreaction=false), GrandCanonicalPoint(0.25,0.1)))
    for (eos, x) in cases
        y = FT.coordinates(x)
        D = length(y)
        th = thermodynamic(eos,x)
        th3 = thermodynamic_perturbation(eos,x)
        @test th3.pressure ≈ th.pressure rtol=1e-10
        @test th3.gradient ≈ th.gradient rtol=1e-10
        @test th3.hessian ≈ th.hessian rtol=1e-10
        scale = maximum(abs, th3.third)
        for k in 1:D
            fd = fd_coordinate(hessian_at(eos), y, k)
            @test th3.third[:,:,k] ≈ fd rtol=1e-5 atol=1e-7*scale
        end
        # fully symmetric, except HeavyQuark without back-reaction, whose composed Hessian is
        # not the Hessian of the pressure (then only the derivative-of-Hessian symmetry holds)
        full_symmetry = !(eos isa HeavyQuark) || FT.backreaction(eos)
        for i in 1:D, j in 1:D, k in 1:D
            @test th3.third[i,j,k] ≈ th3.third[j,i,k] atol=1e-9*scale
            full_symmetry && @test th3.third[i,j,k] ≈ th3.third[k,j,i] atol=1e-9*scale
        end
    end
end

@testset "Third derivatives: algebra, units, tabulated, shorthands" begin
    eos = IdealQCD()
    x = GrandCanonicalPoint(0.2,0.05)
    ref = thermodynamic_perturbation(eos,x)
    for (model, factor) in ((eos+eos,2), (2*eos,2), (eos*3,3), (eos/0.5,2), (-eos,-1), (3*eos-eos,2))
        th3 = thermodynamic_perturbation(model,x)
        @test th3.third ≈ factor*ref.third
        @test th3.hessian ≈ factor*ref.hessian
    end
    converted = convert_units(GeVFm(), eos)
    xc = convert_units(GeVFm(), NaturalGeV(), x)
    th3c = thermodynamic_perturbation(converted, xc)
    f = FT._conversion_factors(NaturalGeV(), GeVFm())
    @test th3c.third ≈ f.third*ref.third
    @test convert_units(NaturalGeV(), th3c).third ≈ ref.third
    @test convert_units(GeVFm(), th3c) === th3c
    yc = FT.coordinates(xc)
    for k in 1:2
        @test th3c.third[:,:,k] ≈ fd_coordinate(hessian_at(converted), yc, k) rtol=1e-5
    end
    tabulated = TabulatedEOS(FluiduMEoS(), range(0.1,0.5; length=50))
    @test_throws ArgumentError thermodynamic_perturbation(tabulated, GrandCanonicalPoint(0.2))
    @test thermodynamic_perturbation(eos,0.2,0.05).third == ref.third
    @test thermodynamic_perturbation(eos,0.2,SVector(0.05)).third == ref.third
    @test @inferred(thermodynamic_perturbation(eos,0.2)).third[1,1,1] ≈ eos.a1*0.2
end

@testset "Transport gradients" begin
    eos = IdealQCD()
    model = TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),ZeroChargeTransport())
    fluid = FluidModel(eos,model)
    # (a distinct name: assigning `tr` inside the closure would rebind the testset's `tr`)
    coefficients(fl) = z -> begin
        t = transport(fl, FT.grandcanonical_from_coordinates(z))[2]
        SVector(t.shear.eta, t.shear.tau, t.bulk.zeta, t.bulk.tau)
    end
    for x in (GrandCanonicalPoint(0.2),GrandCanonicalPoint(0.2,0.1))
        y = FT.coordinates(x)
        D = length(y)
        th3, tr = @inferred transport_perturbation(fluid,x)
        th, tr0 = transport(fluid,x)
        @test th3 isa ThermodynamicPerturbationState
        @test tr.shear isa ShearPerturbationState
        @test tr.bulk isa BulkPerturbationState
        @test tr.shear.eta ≈ tr0.shear.eta
        @test tr.shear.tau ≈ tr0.shear.tau
        @test tr.bulk.zeta ≈ tr0.bulk.zeta
        @test tr.bulk.tau ≈ tr0.bulk.tau
        @test ShearState(tr.shear).tau == tr.shear.tau
        @test BulkState(tr.bulk).zeta == tr.bulk.zeta
        @test length(tr.shear.eta_gradient) == D
        for k in 1:D
            fd = fd_coordinate(coefficients(fluid), y, k)
            @test tr.shear.eta_gradient[k] ≈ fd[1] rtol=1e-6
            @test tr.shear.tau_gradient[k] ≈ fd[2] rtol=1e-6
            @test tr.bulk.zeta_gradient[k] ≈ fd[3] rtol=1e-6
            @test tr.bulk.tau_gradient[k] ≈ fd[4] rtol=1e-6
        end
        # the same with the model-level call and a plain state
        tr1 = transport_perturbation(model, eos, x, th)
        @test tr1.shear.tau_gradient ≈ tr.shear.tau_gradient
        # transport accepts the perturbation state
        @test transport(model, x, th3).shear.tau ≈ tr.shear.tau
    end

    # state-dependent models go through the ForwardDiff fallback
    functional = TransportModel(EtaOverS((x,th) -> 0.08 + 1.1*(x.T-0.155), 5.0),
                                ZetaOverS((x,th) -> 0.03*exp(-(x.T-0.18)^2/0.001), 10.0),
                                ZeroChargeTransport())
    for fluid2 in (FluidModel(FluiduMEoS(),functional), FluidModel(IdealQCD(),functional))
        x = GrandCanonicalPoint(0.2)
        y = FT.coordinates(x)
        th3, tr = transport_perturbation(fluid2,x)
        fd = fd_coordinate(coefficients(fluid2), y, 1)
        @test tr.shear.eta_gradient[1] ≈ fd[1] rtol=1e-6
        @test tr.shear.tau_gradient[1] ≈ fd[2] rtol=1e-6
        @test tr.bulk.zeta_gradient[1] ≈ fd[3] rtol=1e-6
        @test tr.bulk.tau_gradient[1] ≈ fd[4] rtol=1e-6
    end

    zero_model = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport())
    th3, tr = transport_perturbation(FluidModel(eos,zero_model), GrandCanonicalPoint(0.2,0.1))
    @test tr.shear.eta == tr.shear.tau == 0
    @test length(tr.shear.eta_gradient) == 2 && all(iszero, tr.shear.eta_gradient)
    @test length(tr.bulk.tau_gradient) == 2 && all(iszero, tr.bulk.tau_gradient)

    so = TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),ZeroChargeTransport(),
                        ConstantSecondOrderTransport(:test; shear=(a=1.0,)))
    th3, tr = transport_perturbation(FluidModel(eos,so), GrandCanonicalPoint(0.2))
    @test tr.second_order.shear.a == 1.0

    th3, tr = transport_perturbation(fluid, 0.2, 0.1)
    @test tr.shear.eta_gradient ≈ transport_perturbation(fluid, GrandCanonicalPoint(0.2,0.1))[2].shear.eta_gradient
    @test isempty(Test.detect_ambiguities(FluidumThermodynamics))
end
