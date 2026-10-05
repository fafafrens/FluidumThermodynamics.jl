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
