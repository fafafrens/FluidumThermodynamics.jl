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

@testset "Dimensionless charge transport" begin
    eos = IdealQCD()
    hbarc = FluidumThermodynamics._hbarc_GeV_fm
    x = GrandCanonicalPoint(0.2,0.1)
    th = thermodynamic(eos,x)
    chi_a = x.T*susceptibilities(th)                       # ∂n/∂(μ/T)
    for (model, kappa_expected) in ((ConstantKappaOverT2(0.3,0.2), 0.3*x.T^2),
                                    (ConstantDsT(0.2,0.2), 0.2*charge_density(th,1)/x.T))
        tr = transport(TransportModel(ZeroShearTransport(),ZeroBulkTransport(),model),x,th)
        @test size(tr.charge.kappa) == size(tr.charge.tau) == (1,1)
        @test tr.charge.kappa[1,1] ≈ kappa_expected
        # the diffusion front κ/(τ_ν ∂n/∂(μ/T)) is Ctau by construction
        @test tr.charge.kappa[1,1]/(tr.charge.tau[1,1]*chi_a[1,1]) ≈ 0.2
        # a zero-charge point has no charge sector, not an error
        tr0 = transport(FluidModel(eos,TransportModel(ZeroShearTransport(),ZeroBulkTransport(),model)),
                        GrandCanonicalPoint(0.2))[2]
        @test size(tr0.charge.kappa) == (0,0)
    end
    # unit conventions: κ is length⁻², τ is length, the front speed is a pure number
    fm = thermodynamic(convert_units(GeVFm(),eos),x)
    for model in (ConstantKappaOverT2(0.3,0.2), ConstantDsT(0.2,0.2))
        tm = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),model)
        nat, gevfm = transport(tm,x,th), transport(tm,x,fm)
        @test gevfm.charge.kappa[1,1] ≈ nat.charge.kappa[1,1]/hbarc^2
        @test gevfm.charge.tau[1,1] ≈ nat.charge.tau[1,1]*hbarc
        @test gevfm.charge.kappa[1,1]/(gevfm.charge.tau[1,1]*x.T*susceptibilities(fm)[1,1]) ≈ 0.2
    end
    # several charges: diagonal κ, matrix τ_ν with κ·(Ctau·T·χ)⁻¹
    eos2 = PolynomialMultiChargeEOS(3.0, SMatrix{2,2}(1.0,0.2,0.2,0.8))
    x2 = GrandCanonicalPoint(0.2,0.05,-0.03)
    th2 = thermodynamic(eos2,x2)
    tr2 = transport(TransportModel(ZeroShearTransport(),ZeroBulkTransport(),ConstantKappaOverT2(0.3,0.2)),x2,th2)
    @test tr2.charge.kappa ≈ 0.3*x2.T^2*I
    @test tr2.charge.tau*(0.2*x2.T*susceptibilities(th2)) ≈ tr2.charge.kappa
end

function functional_allocations(fluid, x)
    transport(fluid,x)
    @allocated(transport(fluid,x))
end

@testset "State-dependent shear and bulk" begin
    eos = FluidumEOS()
    x = GrandCanonicalPoint(0.2)
    th = thermodynamic(eos,x)
    s, w = entropy_density(th), enthalpy(x,th)
    ηs = (x,th) -> 0.08 + 1.1*(x.T-0.155)
    Cτ = (x,th) -> 5.0 + 10*x.T
    ζs = (x,th) -> 0.03/(1+((x.T-0.175)/0.024)^2)
    # Numbers give exactly the constant models.
    same = transport(TransportModel(EtaOverS(0.16,5.0),ZetaOverS(0.03,10.0),ZeroChargeTransport()),x,th)
    ref = transport(TransportModel(ConstantEtaOverS(0.16,5.0),ConstantZetaOverS(0.03,10.0),ZeroChargeTransport()),x,th)
    @test same.shear == ref.shear && same.bulk == ref.bulk
    @test EtaOverS(0.16,5.0).eta_over_s(x,th) == 0.16
    # Mixed number types promote in the constant models.
    @test ConstantEtaOverS(0.16,5) == ConstantEtaOverS(0.16,5.0)
    @test transport(TransportModel(EtaOverS(0.16,5),ZetaOverS(3//100,10),ZeroChargeTransport()),x,th).shear == ref.shear
    # Callables see the point and the state.
    tr = transport(TransportModel(EtaOverS(ηs,Cτ),ZetaOverS(ζs,10),ZeroChargeTransport()),x,th)
    @test tr.shear.eta ≈ ηs(x,th)*s
    @test tr.shear.tau ≈ tr.shear.eta/(Cτ(x,th)*w)
    @test tr.bulk.zeta ≈ ζs(x,th)*s
    @test tr.bulk.tau ≈ tr.bulk.zeta/(10*w)
    x2 = GrandCanonicalPoint(0.3)
    th2 = thermodynamic(eos,x2)
    tr2 = transport(TransportModel(EtaOverS(ηs,Cτ),ZetaOverS(ζs,10),ZeroChargeTransport()),x2,th2)
    @test tr2.shear.eta/entropy_density(th2) ≈ ηs(x2,th2)
    linear = FluidModel(eos,TransportModel(EtaOverS(ηs,Cτ),ZetaOverS(ζs,10.0),ZeroChargeTransport()))
    @inferred transport(linear,x)
    @test functional_allocations(linear,x) == 0
    # Dependence on the chemical potential and on the state itself.
    qcd = IdealQCD()
    y = GrandCanonicalPoint(0.2,0.3)
    thy = thermodynamic(qcd,y)
    ημ = (x,th) -> 0.1*(1 + (x.μ[1]/x.T)^2)
    Cs = (x,th) -> 5.0*entropy_density(th)/charge_density(th,1)
    try_ = transport(TransportModel(EtaOverS(ημ,Cs),ZeroBulkTransport(),ZeroChargeTransport()),y,thy)
    @test try_.shear.eta ≈ 0.1*(1+(0.3/0.2)^2)*entropy_density(thy)
    @test try_.shear.tau ≈ try_.shear.eta/(Cs(y,thy)*enthalpy(y,thy))
    charged = FluidModel(qcd,TransportModel(EtaOverS(ημ,Cs),ZetaOverS(ζs,10.0),ZeroChargeTransport()))
    @inferred transport(charged,y)
    @test functional_allocations(charged,y) == 0
    # Functional sectors return the state directly, e.g. a Fluidum-like bulk
    # relaxation time with the speed of sound and a floor.
    shear = FunctionalShearTransport((x,th) -> ShearState(0.16*entropy_density(th),0.5))
    # (`ent`, not `s`: assigning to a name of the enclosing scope would box it.)
    bulk = FunctionalBulkTransport((x,th) -> begin
        ent = entropy_density(th)
        ζ = ζs(x,th)*ent
        BulkState(ζ,ζ/(10*x.T*ent)/(1/3-speed_of_sound_squared(x,th))^2 + 0.5)
    end)
    fluid = FluidModel(eos,TransportModel(shear,bulk,ZeroChargeTransport()))
    thf, trf = @inferred transport(fluid,x)
    @test trf.shear.eta ≈ 0.16*s && trf.shear.tau == 0.5
    @test trf.bulk.zeta ≈ ζs(x,th)*s
    @test trf.bulk.tau ≈ trf.bulk.zeta/(10*x.T*s)/(1/3-speed_of_sound_squared(x,th))^2 + 0.5
    @test functional_allocations(fluid,x) == 0
    @test_throws ArgumentError transport(FluidModel(eos,TransportModel(FunctionalShearTransport((x,th) -> (1.0,2.0)),ZeroBulkTransport(),ZeroChargeTransport())),x)
    @test_throws ArgumentError transport(FluidModel(eos,TransportModel(ZeroShearTransport(),FunctionalBulkTransport((x,th) -> nothing),ZeroChargeTransport())),x)
end

@testset "State-dependent diffusion" begin
    eos = IdealQCD()
    x = GrandCanonicalPoint(0.2,0.1)
    th = thermodynamic(eos,x)
    zero_sectors(charge) = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),charge)
    # Numbers give exactly ConstantDsT.
    same = transport(zero_sectors(EinsteinDiffusion(0.2,1.0)),x,th)
    ref = transport(zero_sectors(ConstantDsT(0.2,1.0)),x,th)
    @test same.charge == ref.charge
    # Fluidum's LinearDiffusion: DₛT linear in T above Tc.
    linear = (x,th) -> ifelse(x.T > 0.18, 1.2*x.T + 0.05, 1.2*0.18 + 0.05)
    tr = transport(zero_sectors(EinsteinDiffusion(linear,1.0)),x,th)
    @test tr.charge == transport(zero_sectors(ConstantDsT(linear(x,th),1.0)),x,th).charge
    fluid = FluidModel(eos,zero_sectors(EinsteinDiffusion(linear,(x,th) -> 0.5 + x.T)))
    @inferred transport(fluid,x)
    @test functional_allocations(fluid,x) == 0
    # Fluidum's free heavy-quark relaxation time as a state-dependent Ctau on a
    # Boltzmann gas, where T·∂n/∂μ = n: τ = DₛT z³/(48T) (2K₁-3K₃+K₅)/K₂.
    m, DsT = 1.5, 0.2
    hq = HadronResonanceGas([HadronSpecies(m,6,(1,))])
    y = GrandCanonicalPoint(0.2,0.03)
    bk = (ν,z) -> FluidumThermodynamics.besselkx(ν,z)
    charm_Ctau = (x,th) -> begin
        zz = m/x.T
        48*bk(2,zz)/(zz^3*(2*bk(1,zz)-3*bk(3,zz)+bk(5,zz)))
    end
    charm = FluidModel(hq,zero_sectors(EinsteinDiffusion(DsT,charm_Ctau)))
    thc, trc = @inferred transport(charm,y)
    z = m/y.T
    @test trc.charge.kappa[1,1] ≈ DsT/y.T*charge_density(thc,1)
    @test trc.charge.tau[1,1] ≈ DsT*z^3/(48*y.T)*(2*bk(1,z)-3*bk(3,z)+bk(5,z))/bk(2,z)
end

# A callable struct that closes over the charm hadron list, from the example.
include(joinpath(@__DIR__,"..","examples","hadronic_charm_diffusion.jl"))

@testset "Callable structs close over data" begin
    DsT = 0.2
    zero_sectors(charge) = TransportModel(ZeroShearTransport(),ZeroBulkTransport(),charge)
    bk = (ν,z) -> FluidumThermodynamics.besselkx(ν,z)
    τ̂(m,T) = (z = m/T; z^3/(48*T)*(2*bk(1,z)-3*bk(3,z)+bk(5,z))/bk(2,z))
    y = GrandCanonicalPoint(0.2,0.03)
    # One species: Fluidum's free heavy-quark relaxation time.
    quark = [HadronSpecies(1.5,6,(1,))]
    one = FluidModel(HadronResonanceGas(quark),zero_sectors(EinsteinDiffusion(DsT,CharmHadronCtau(quark))))
    th1, tr1 = @inferred transport(one,y)
    @test tr1.charge.tau[1,1] ≈ DsT*τ̂(1.5,y.T)
    @test tr1.charge.kappa[1,1] ≈ DsT/y.T*charge_density(th1,1)
    # Several hadrons, including a q = 2 state: κ from the total charm density and
    # τ the χ-weighted mean of the single-species values, as in Fluidum.
    hadrons = [HadronSpecies(1.87,2,(1,)),HadronSpecies(2.29,2,(1,)),HadronSpecies(3.1,3,(2,))]
    eos = HadronResonanceGas(hadrons)
    th = thermodynamic(eos,y)
    tr = transport(zero_sectors(EinsteinDiffusion(DsT,CharmHadronCtau(hadrons))),y,th)
    qs = [s.charges[1] for s in hadrons]
    ns = [charge_density(thermodynamic(HadronResonanceGas([s]),y),1) for s in hadrons]./qs
    @test y.T*susceptibility(th,1,1) ≈ sum(qs.^2 .* ns)
    @test tr.charge.kappa[1,1] ≈ DsT/y.T*charge_density(th,1)
    @test tr.charge.tau[1,1] ≈ DsT*sum(qs.^2 .* ns .* [τ̂(s.mass,y.T) for s in hadrons])/sum(qs.^2 .* ns)
    @inferred transport(FluidModel(eos,zero_sectors(EinsteinDiffusion(DsT,CharmHadronCtau(hadrons)))),y)
end
