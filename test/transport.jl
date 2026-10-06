@testset "Transport sectors" begin
    eos = IdealQCD()
    zero_model = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ZeroChargeTransport())
    for x in (GrandCanonicalPoint(0.2),GrandCanonicalPoint(0.2,0.1))
        th, tr = transport(FluidModel(eos,zero_model),x)
        @test tr.shear.eta == tr.shear.tau == 0
        @test tr.bulk.zeta == tr.bulk.tau == 0
        @test size(tr.charge.kappa) == (length(x.μ),length(x.μ))
        @test all(iszero,tr.charge.kappa)
        @test all(iszero,tr.charge.tau)
        @test eltype(tr.charge.kappa) == eltype(tr.charge.tau) == typeof(th.pressure)
        model = TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),ZeroChargeTransport())
        tr = transport(model,x,th)
        w = energy_density(x,th)+th.pressure
        @test tr.shear.eta ≈ 0.16*entropy_density(th)
        @test tr.shear.tau ≈ tr.shear.eta/(5*w)
        @test tr.bulk.zeta ≈ 0.03*entropy_density(th)
        @test tr.bulk.tau ≈ tr.bulk.zeta/(10*w)
    end
    wrong = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),
                           ConstantChargeTransport(ones(2,2),ones(2,2)))
    @test_throws DimensionMismatch transport(FluidModel(eos,wrong),GrandCanonicalPoint(0.2,0.0))
end

@testset "Transport from plain numbers" begin
    eos = IdealQCD()
    model = TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),ZeroChargeTransport())
    fluid = FluidModel(eos,model)
    for (args, x) in (((0.2,), GrandCanonicalPoint(0.2)),
                      ((0.2,0.1), GrandCanonicalPoint(0.2,0.1)),
                      ((0.2,SVector(0.1)), GrandCanonicalPoint(0.2,0.1)))
        th, tr = transport(fluid,args...)
        th0, tr0 = transport(fluid,x)
        @test th.pressure == th0.pressure
        @test th.gradient == th0.gradient
        @test tr.shear.eta == tr0.shear.eta
        @test tr.bulk.tau == tr0.bulk.tau
        @test size(tr.charge.kappa) == size(tr0.charge.kappa)
    end
end


function plain_matrix_allocations(eos, fluid, x)
    thermodynamic(eos,x); transport(fluid,x)
    @allocated(thermodynamic(eos,x)), @allocated(transport(fluid,x))
end

@testset "Plain matrices are stored as static matrices" begin
    χ = [2.0 0.3; 0.3 1.5]
    κ, τ = [0.4 0.05; 0.05 0.3], [1.0 0.0; 0.0 1.2]
    plain = PolynomialMultiChargeEOS(3.0,χ)
    static = PolynomialMultiChargeEOS(3.0,SMatrix{2,2}(χ))
    @test plain.chi isa SMatrix{2,2,Float64}
    @test typeof(plain) == typeof(static)
    model = ConstantChargeTransport(κ,τ)
    @test model.kappa isa SMatrix{2,2,Float64} && model.tau isa SMatrix{2,2,Float64}
    @test model.kappa == κ && model.tau == τ
    # The model owns a copy: changing the input afterwards has no effect.
    χ[1,1] = 99.0
    @test plain.chi[1,1] == 2.0
    x = GrandCanonicalPoint(0.2,0.03,-0.01)
    @test thermodynamic(plain,x).hessian == thermodynamic(static,x).hessian
    fluid = FluidModel(plain,TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),model))
    plain_matrix_allocations(plain,fluid,x)
    @test all(iszero, plain_matrix_allocations(plain,fluid,x))
    # Static input is kept as is, and large matrices are not converted.
    S = @SMatrix [1.0 0.0; 0.0 1.0]
    @test ConstantChargeTransport(S,S).kappa === S
    @test ConstantChargeTransport(ones(9,9),ones(9,9)).kappa isa Matrix
end
