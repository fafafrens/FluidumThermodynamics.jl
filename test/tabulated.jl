@testset "Tabulated EOS" begin
    eos = FluidumEOS()
    nodes = range(0.08,0.6,length=100)
    table = TabulatedEOS(eos,nodes)
    @test unit_system(table) == NaturalGeV()
    # Quintic Hermite of p/T⁴ converges like h⁶ with exact ∂s/∂T.
    worst = zeros(3)
    for T in range(0.0801,0.5999,length=1001)
        a, b = thermodynamic(table,T), thermodynamic(eos,T)
        worst .= max.(worst, abs.((a.pressure/b.pressure-1, a.gradient[1]/b.gradient[1]-1,
                                   a.hessian[1]/b.hessian[1]-1)))
    end
    @test worst[1] < 1e-8
    @test worst[2] < 1e-7
    @test worst[3] < 1e-5
    # Nodes are reproduced exactly, and both sides of a node agree.
    for T in nodes[[1,37,100]]
        @test thermodynamic(table,T).pressure ≈ thermodynamic(eos,T).pressure rtol=1e-14
        @test thermodynamic(table,T).hessian[1] ≈ thermodynamic(eos,T).hessian[1] rtol=1e-12
    end
    T = nodes[50]
    left, right = thermodynamic(table,prevfloat(T)), thermodynamic(table,nextfloat(T))
    @test left.hessian[1] ≈ right.hessian[1] rtol=1e-10
    # The table is consistent: s and ∂s/∂T are derivatives of the interpolated pressure.
    x = GrandCanonicalPoint(0.2345)
    @test ForwardDiff.derivative(t -> pressure(table,GrandCanonicalPoint(t)),0.2345) ≈ entropy_density(thermodynamic(table,x)) rtol=1e-12
    @test ForwardDiff.derivative(t -> entropy_density(thermodynamic(table,GrandCanonicalPoint(t))),0.2345) ≈ thermodynamic(table,x).hessian[1] rtol=1e-12
    @test pressure(table,x) == thermodynamic(table,x).pressure
    @inferred thermodynamic(table,x)
    # A conformal EOS is exact even on a coarse table.
    conformal = TabulatedEOS(IdealQCD(),range(0.1,0.5,length=3))
    @test thermodynamic(conformal,0.2345).pressure ≈ thermodynamic(IdealQCD(),0.2345).pressure rtol=1e-14
    @test speed_of_sound_squared(x,thermodynamic(conformal,x)) ≈ 1/3 rtol=1e-12
    # Microcanonical inversion needs no guess.
    for T in (0.08,0.1234,0.3,0.6)
        th = thermodynamic(table,T)
        ε = energy_density(GrandCanonicalPoint(T),th)
        @test grandcanonical(table,MicrocanonicalPoint(ε)).T ≈ T rtol=1e-14
        @test thermodynamic(table,MicrocanonicalPoint(ε)).pressure ≈ th.pressure rtol=1e-13
    end
    # Estimating ∂s/∂T from p and s alone is less accurate but consistent.
    temperatures = collect(nodes)
    states = [thermodynamic(eos,t) for t in temperatures]
    estimated = TabulatedEOS(temperatures,[th.pressure for th in states],[entropy_density(th) for th in states])
    @test thermodynamic(estimated,0.2345).pressure ≈ thermodynamic(eos,0.2345).pressure rtol=1e-6
    @test thermodynamic(estimated,0.2345).hessian[1] ≈ thermodynamic(eos,0.2345).hessian[1] rtol=1e-3
    # Unit conventions follow the sampled EOS.
    mev = TabulatedEOS(FluidumEOS(units=MeVFm()),1000 .* nodes)
    @test unit_system(mev) == MeVFm()
    converted = convert_units(MeVFm(),thermodynamic(table,0.2345))
    @test thermodynamic(mev,234.5).pressure ≈ converted.pressure rtol=1e-8
    @test thermodynamic(mev,234.5).hessian[1] ≈ converted.hessian[1] rtol=1e-5
    # Invalid tables and points.
    @test_throws DomainError thermodynamic(table,0.07)
    @test_throws DomainError thermodynamic(table,0.61)
    @test_throws DomainError grandcanonical(table,MicrocanonicalPoint(1e3))
    @test_throws DimensionMismatch thermodynamic(table,GrandCanonicalPoint(0.2,0.0))
    @test_throws ArgumentError TabulatedEOS([0.2,0.1],[1.0,2.0],[1.0,1.0],[1.0,1.0])
    @test_throws ArgumentError TabulatedEOS([0.1,0.2],[1.0,2.0],[1.0,1.0],[1.0,-1.0])
    @test_throws ArgumentError TabulatedEOS([0.1],[1.0],[1.0],[1.0])
    @test_throws DimensionMismatch TabulatedEOS([0.1,0.2],[1.0],[1.0,1.0],[1.0,1.0])
    @test_throws ArgumentError TabulatedEOS([0.1,0.2,0.3],[1.0,2.0,3.0],[1.0,1.0,1.0])
end
