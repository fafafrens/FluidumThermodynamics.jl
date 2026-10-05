# Second-order reference from Fluidum.jl, commit 4fdd25fc311d64619d8b2f2af44054775dbec50f.
# src/EquationOfState/FluiduMEoS.jl, MIT © 2024 Eduardo Grossi.
# Third-order routines omitted; @muladd annotations removed to keep tests dependency-free.
struct FluiduMEoS{T} <:EquationOfState
    a1::T
    a2::T
    a3::T
    a4::T
    b1::T
    b2::T
    b3::T
    b4::T
    c::T
    d::T

end

function FluiduMEoS()

    a1 = -15.526548963383643
    a2 = 18.6159584620131
    b1 = -3.3147904483107595
    b2 = 5.310983721567554
    a3 = -10.731808109698516
    a4 = 2.7413302179949346
    b4 = 1.8600649533271152
    b3 = -4.653922019495976
    c = -1.0465330501411811
    d = 0.09551531822245873
    FluiduMEoS(a1,a2,a3,a4,b1,b2,b3,b4,c,d)
end

@inline @fastmath function den(t,b1,b2,b3,b4)
    invt=1/t
    x=154*invt

    invt2=invt*invt
    invt3=invt*invt2
    fun=@evalpoly(x,1,b1,b2,b3,b4)
    fun1=@evalpoly(x,b1,2*b2,3*b3,4*b4)
    fun2=@evalpoly(x,2*b2,6*b3,12*b4)

    (fun,(-154*invt2*fun1) ,(308*invt3*fun1+(23716)*fun2*invt3*invt))
end

@inline @fastmath function num(t,a1,a2,a3,a4)

    invt=1/t
    x=154*invt
    invt2=invt*invt
    invt3=invt*invt2
    fun=@evalpoly(x,5.208957878352717,a1,a2,a3,a4)
    fun1=@evalpoly(x,a1,2*a2,3*a3,4*a4)
    fun2=@evalpoly(x,2*a2,6*a3,12*a4)
    (fun,(-154*invt2*fun1) ,(308*invt3*fun1+(23716)*fun2*invt3*invt))
end

@inline @fastmath function exponent(t,c,d)

    c1=-c^2
    d1=-d^2
    invt=1/t
    x=100*invt
    invt2=invt*invt
    invt3=invt*invt2
    fac1=exp(muladd(d1,x,c1)*x)
    tinypol=muladd(x,2*d1,c1)

    der=fac1*tinypol
    der2=muladd(tinypol,der ,fac1*(2*d1))

    (fac1,(-100*invt2*der) ,(200*invt3*der+(10_000)*der2*invt3*invt))

end

@inline @fastmath function thermodynamic(T,x::FluiduMEoS)

    t=T*1000
    (d,d1,d2)=den(t,x.b1,x.b2,x.b3,x.b4)
    (n,n1,n2)=num(t,x.a1,x.a2,x.a3,x.a4)
    (e,e1,e2)=exponent(t,x.c,x.d)
    invd=1/d
    invd2=invd*invd
    invd3=invd2*invd
     fun=e*n*invd*fmGeV3
     fun1=(d*n*e1+e*(d*n1-n*d1))*invd2*fmGeV3

     fun2=(d*(2*e1*(-n*d1+d*n1)+d*n*e2)+e*(n*(2*d1*d1-d*d2)+d*(-2*d1*n1+d*n2)))*invd3*fmGeV3
    T2=T*T
    T3=T2*T
    zero_d=T3*T*fun
     one_d=T3*(4*fun+1_000*T*fun1)
     two_d=T2*(12*fun+T*(8_000*fun1+1_000_000*T*fun2))

    Thermodynamic(zero_d,(one_d,),(two_d,))

end
