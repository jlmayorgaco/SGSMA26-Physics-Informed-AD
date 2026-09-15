# Analytic second-order descriptor sensitivities for the frozen IEEE39 DAE.
# Directional Hessian contractions use ForwardDiff on the compiled residual.
# Frozen dictionaries and likelihoods are read only.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII, ForwardDiff
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra, CSV, DataFrames, Random, DelimitedFiles, Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "analytic_second_order_dae_v1")
const RES = joinpath(OUT, "results")
const META = joinpath(OUT, "metadata")
mkpath(RES); mkpath(META)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const BUSES = [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
const OBS = [2,5,6,10,19,22,29,39]
const EDGE_FOR = [11,37,18,35,31,1,8,5]
const NATIVE_FOR = [false,false,true,true,true,false,false,false]
const DT = 1/30
const NF = 30

function build()
    mod = Module(Symbol("SECOND_ORDER_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve))
    redirect_stdout(devnull) do
        Base.include_string(mod, read(SRC, String), "second_order_ieee39.jl")
    end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pfs, verbose=false)
    return nw, s0
end

function output_symbols()
    out = Any[]
    for b in OBS
        push!(out, VIndex(b, :busbar₊u_r)); push!(out, VIndex(b, :busbar₊u_i))
    end
    for (e,native) in zip(EDGE_FOR, NATIVE_FOR)
        side = native ? :src : :dst
        push!(out, EIndex(e, Symbol(side, "₊i_r"))); push!(out, EIndex(e, Symbol(side, "₊i_i")))
    end
    out
end

nw, s0 = build()
x0 = Float64.(uflat(s0)); p0 = Float64.(pflat(s0)); n = length(x0); np = length(p0)
M = Matrix(nw.mass_matrix); md = diag(M)
didx = findall(!iszero, md); aidx = findall(iszero, md)
nd, na = length(didx), length(aidx)
syms = SII.variable_symbols(nw); ps = SII.parameter_symbols(nw)

residual(x,p) = begin
    du = zeros(eltype(x), n)
    nw(du, x, p, 0.0)
    du
end
outsym = output_symbols()
# The frozen PMU contract uses affine PiLine terminal-current channels, not
# the internal dynamic edge wrapper.  Construct that map explicitly so its
# parameter direct term and exact measurement Hessian are both transparent.
state_index(sym::String) = only(findall(s -> string(s) == sym, syms))
vri = [state_index("VIndex($(b), :busbar₊u_r)") for b in 1:39]
vii = [state_index("VIndex($(b), :busbar₊u_i)") for b in 1:39]
function pmu_coefficients()
    data = CSV.read(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data", "branch.csv"), DataFrame)
    out = Tuple{Int,Int,ComplexF64,ComplexF64,Bool}[]
    for (edge,native) in zip(EDGE_FOR,NATIVE_FOR)
        row = data[edge,:]; i = Int(row.src_bus); j = Int(row.dst_bus)
        ys = 1 / complex(Float64(row.R), Float64(row.X)); tap = Float64(row.r_src) == 0 ? 1.0 : Float64(row.r_src)
        a = -(ys + complex(Float64(row.G_src), Float64(row.B_src))) * tap^2; b = ys*tap
        c = ys*tap; d = -(ys + complex(Float64(row.G_dst), Float64(row.B_dst)))
        push!(out,(i,j,native ? a : c,native ? b : d,native))
    end
    out
end
const PMU_COEFF = pmu_coefficients()
obsfun(x,p) = begin
    T = promote_type(eltype(x),eltype(p)); y = zeros(T, length(outsym)); r = 1
    for b in OBS
        y[r] = x[vri[b]]; y[r+1] = x[vii[b]]; r += 2
    end
    for (i,j,a,b,_) in PMU_COEFF
        vr = a.re*x[vri[i]] - a.im*x[vii[i]] + b.re*x[vri[j]] - b.im*x[vii[j]]
        vi = a.re*x[vii[i]] + a.im*x[vri[i]] + b.re*x[vii[j]] + b.im*x[vri[j]]
        y[r] = vr; y[r+1] = vi; r += 2
    end
    y
end

println("building first-order descriptor Jacobian n=",n," np=",np)
# Use the package's native linearization for the first-order descriptor path.
# Direct parameter Duals through a compiled NetworkDynamics graph are not
# type-stable because the graph caches Float64 output buffers; the supported
# linearize_network API performs the same AD with the correct coordinate types.
desc = linearize_network(s0)
A = Matrix(desc.A)
A_dd = A[didx,didx]; A_da = A[didx,aidx]; A_ad = A[aidx,didx]; A_aa = A[aidx,aidx]
rank(A_aa) == na || error("algebraic Jacobian rank deficient")
As = A_dd - A_da * (A_aa \ A_ad)

Pdir = zeros(Float64,np,length(BUSES)); Bfull = zeros(Float64,n,length(BUSES)); Dsrc = zeros(Float64,length(outsym),length(BUSES)); Cfull = zeros(Float64,length(outsym),n)
pmap = DataFrame(candidate_bus=Int[],p_symbol=String[],q_symbol=String[],p_index=Int[],q_index=Int[],p_value=Float64[],q_value=Float64[],p_second_derivative=Float64[],q_second_derivative=Float64[])
for (k,bus) in enumerate(BUSES)
    psym = VIndex(bus,:ZIPLoad₊Pset); qsym = VIndex(bus,:ZIPLoad₊Qset)
    ip = only(SII.parameter_index(nw,psym)); iq = only(SII.parameter_index(nw,qsym))
    Pdir[ip,k] = p0[ip]; Pdir[iq,k] = p0[iq]
    # Parameter direction is the actual ZIP event p=p0*(1+a), q=q0*(1+a).
    # Obtain both state residual and PMU feedthrough through native descriptor
    # linearization, then combine the P/Q columns in physical units.
    lstate = linearize_network(s0; in=[psym, qsym], out=syms)
    Bfull[:,k] = Matrix(lstate.B) * [p0[ip], p0[iq]]
    Dsrc[:,k] = zeros(Float64,length(outsym))
    if k == 1
        Cfull .= ForwardDiff.jacobian(x -> obsfun(x,p0), x0)
    end
    push!(pmap,(bus,string(psym),string(qsym),ip,iq,p0[ip],p0[iq],0.0,0.0))
end
CSV.write(joinpath(RES,"event_parameter_map.csv"),pmap)
writedlm(joinpath(RES,"C_pmu_frozen.csv"), Cfull, ',')
CSV.write(joinpath(RES,"measurement_parameter_audit.csv"),
    DataFrame(candidate_bus=BUSES, direct_D_norm=[norm(Dsrc[:,k]) for k in 1:length(BUSES)],
              direct_D_max=[maximum(abs.(Dsrc[:,k])) for k in 1:length(BUSES)]))
let hfd = 1e-5
    fdrows = DataFrame(candidate_bus=Int[], fd_rel_error=Float64[], fd_norm=Float64[], ad_norm=Float64[])
    for k in 1:length(BUSES)
        fp = obsfun(x0, p0 + hfd*Pdir[:,k]); fm = obsfun(x0, p0 - hfd*Pdir[:,k])
        fd = (fp-fm)/(2hfd); push!(fdrows,(BUSES[k],norm(fd-Dsrc[:,k])/max(norm(fd),1e-300),norm(fd),norm(Dsrc[:,k])))
    end
    CSV.write(joinpath(RES,"measurement_parameter_fd_audit.csv"),fdrows)
end
CSV.write(joinpath(META,"state_order.csv"),DataFrame(index=1:n,symbol=string.(syms),mass=md,kind=[iszero(x) ? "algebraic" : "differential" for x in md]))
CSV.write(joinpath(META,"parameter_order.csv"),DataFrame(index=1:np,symbol=string.(ps)))
writedlm(joinpath(META,"mass_matrix.csv"),M,','); writedlm(joinpath(META,"A_full.csv"),A,',')

Ba = Bfull[aidx,:]; Bs = Bfull[didx,:] - A_da * (A_aa \ Ba)
aug = zeros(Float64,nd+length(BUSES),nd+length(BUSES)); aug[1:nd,1:nd] .= As; aug[1:nd,nd+1:end] .= Bs
function first_state(t)
    E = exp(aug*t); xd = E[1:nd,nd+1:end]
    xa = -(A_aa \ (A_ad*xd + Ba))
    xf = zeros(Float64,n,length(BUSES)); xf[didx,:] .= xd; xf[aidx,:] .= xa
    xf
end
function full_state(xd, xa)
    xf = zeros(Float64,n); xf[didx] .= xd; xf[aidx] .= xa; xf
end

function projected_hessian(Vxp)
    wbase = vcat(x0,p0)
    fproj(th) = residual(wbase[1:n] + Vxp[1:n,:]*th, wbase[n+1:end] + Vxp[n+1:end,:]*th)
    jproj(th) = ForwardDiff.jacobian(fproj,th)
    Hflat = ForwardDiff.jacobian(jproj,zeros(Float64,length(BUSES)))
    H = zeros(Float64,n,length(BUSES),length(BUSES))
    for l in 1:length(BUSES), m in 1:length(BUSES), k in 1:n
        H[k,l,m] = Hflat[k+(l-1)*n,m]
    end
    for l in 1:length(BUSES), m in l+1:length(BUSES)
        q = (H[:,l,m]+H[:,m,l])/2; H[:,l,m] .= q; H[:,m,l] .= q
    end
    H
end

# Directional Hessian of the PMU output map, including any direct parameter
# dependence. This is the measurement part of the second-order chain rule.
function projected_obs_hessian(Vxp)
    wbase = vcat(x0,p0)
    oproj(th) = obsfun(wbase[1:n] + Vxp[1:n,:]*th,
                       wbase[n+1:end] + Vxp[n+1:end,:]*th)
    jproj(th) = ForwardDiff.jacobian(oproj,th)
    Hflat = ForwardDiff.jacobian(jproj,zeros(Float64,length(BUSES)))
    H = zeros(Float64,nout,length(BUSES),length(BUSES))
    for l in 1:length(BUSES), m in 1:length(BUSES), k in 1:nout
        H[k,l,m] = Hflat[k+(l-1)*nout,m]
    end
    for l in 1:length(BUSES), m in l+1:length(BUSES)
        q = (H[:,l,m]+H[:,m,l])/2; H[:,l,m] .= q; H[:,m,l] .= q
    end
    H
end

pairlist = [(BUSES[i],BUSES[j],i,j) for i in 1:length(BUSES)-1 for j in i+1:length(BUSES)]
npair = length(pairlist); nout = length(outsym)
qself = zeros(Float64,NF*nout,length(BUSES)); qcross = zeros(Float64,NF*nout,npair); danalytic = zeros(Float64,NF*nout,length(BUSES))
qvoltage = zeros(Float64,NF*78,length(BUSES))
symrows = DataFrame(time_s=Float64[],hessian_norm=Float64[],symmetry_max=Float64[],finite=Bool[])
subrows = DataFrame(time_s=Float64[],dynamic_hessian_norm=Float64[],algebraic_hessian_norm=Float64[],dynamic_fraction=Float64[])
onset = DataFrame(candidate_bus=Int[],x1_diff_norm=Float64[],z1_jump_norm=Float64[],x2_diff_norm=Float64[],z2_jump_norm=Float64[],p_second_norm=Float64[])

GammaAug = zeros(Float64,2*nd,2*nd); GammaAug[1:nd,1:nd] .= As; GammaAug[1:nd,nd+1:end] .= Matrix(I,nd,nd)
EG = exp(GammaAug*DT); E = EG[1:nd,1:nd]; G0 = EG[1:nd,nd+1:end]
x2 = zeros(Float64,nd,npair)
x2self = zeros(Float64,nd,length(BUSES))
# Onset derivatives use the same exact algebraic constraint at t=0.  The
# parameter direction has no second derivative because p(a)=p0*(1+a).
V0 = zeros(Float64,n+np,length(BUSES)); V0[n+1:end,:] .= Pdir
V0[aidx,:] .= -(A_aa \ Ba)
H0 = projected_hessian(V0)
Hobs0 = projected_obs_hessian(V0)
for k in 1:NF
    t = (k-1)*DT; xf = first_state(t)
    # Frozen 30-frame contract uses the first sample as the pre-event
    # baseline. Direct feedthrough/algebraic jumps enter post-event.
    if k > 1
        for b in 1:length(BUSES)
            danalytic[(k-1)*nout+1:k*nout,b] .= Cfull*xf[:,b] + Dsrc[:,b]
        end
    end
    if k < NF
        tm = (k-0.5)*DT; xfm = first_state(tm)
        Vxp = zeros(Float64,n+np,length(BUSES)); Vxp[1:n,:] .= xfm; Vxp[n+1:end,:] .= Pdir
        t0h = time(); H = projected_hessian(Vxp); elapsed = time()-t0h
        if k == 1
            # Independent centered check of one residual Hessian contraction
            # at the first midpoint, stored alongside the symmetry audit.
            hh = 1e-5; bchk = 1; vv = Vxp[:,bchk]
            fp = residual(x0 + hh*vv[1:n], p0 + hh*vv[n+1:end]); f0 = residual(x0,p0)
            fm = residual(x0 - hh*vv[1:n], p0 - hh*vv[n+1:end])
            hfd = (fp - 2*f0 + fm)/(hh^2)
            push!(symrows,(tm,norm(H),norm(H[:,bchk,bchk]-hfd),all(isfinite,H)))
        end
        psi = zeros(Float64,n,npair)
        for (r,(_,_,ii,jj)) in enumerate(pairlist); psi[:,r] .= H[:,ii,jj]; end
        psi_self = zeros(Float64,n,length(BUSES))
        for b in 1:length(BUSES); psi_self[:,b] .= H[:,b,b]; end
        b2 = psi[didx,:] - A_da*(A_aa \ psi[aidx,:]); x2 .= E*x2 + G0*b2
        b2self = psi_self[didx,:] - A_da*(A_aa \ psi_self[aidx,:]); x2self .= E*x2self + G0*b2self
        maxsym = 0.0
        for i in 1:length(BUSES), j in i+1:length(BUSES); maxsym=max(maxsym,norm(H[:,i,j]-H[:,j,i])); end
        push!(symrows,(tm,norm(H),maxsym,all(isfinite,H)))
        dn = norm(H[didx,:,:]); an = norm(H[aidx,:,:]); push!(subrows,(tm,dn,an,dn/max(dn+an,1e-300)))
        println(@sprintf("second-order Hessian %d/%d t=%.4f elapsed=%.2fs",k,NF-1,tm,elapsed))
    end
    if k == 1
        z1 = -(A_aa \ Ba)
        for b in 1:length(BUSES)
            z2 = -(A_aa \ H0[aidx,b,b])
            push!(onset,(BUSES[b],0.0,norm(z1[:,b]),norm(x2self[:,b]),norm(z2),0.0))
            qself[1:nout,b] .= 0.0
        end
        for (r,(_,_,ii,jj)) in enumerate(pairlist)
            qcross[1:nout,r] .= 0.0
        end
    end
    if k > 1
        x2a = -(A_aa \ (A_ad*x2))
        x2aself = -(A_aa \ (A_ad*x2self))
        # Complete output second-order chain rule.  The Hessian term is
        # evaluated at the nominal state along each first-order direction.
        Vxe = zeros(Float64,n+np,length(BUSES)); Vxe[1:n,:] .= xf; Vxe[n+1:end,:] .= Pdir
        He = projected_hessian(Vxe)
        Hobse = projected_obs_hessian(Vxe)
        for b in 1:length(BUSES)
            x2aself[:,b] .= -(A_aa \ (A_ad*x2self[:,b] + He[aidx,b,b]))
            qself[(k-1)*nout+1:k*nout,b] .= 0.5 .* (Cfull*full_state(x2self[:,b],x2aself[:,b]) + Hobse[:,b,b])
            vv = zeros(Float64,78)
            for bus in 1:39
                ri = findfirst(==(vri[bus]), aidx); qi = findfirst(==(vii[bus]), aidx)
                vv[2bus-1] = isnothing(ri) ? 0.0 : x2aself[ri,b]
                vv[2bus] = isnothing(qi) ? 0.0 : x2aself[qi,b]
            end
            qvoltage[(k-1)*78+1:k*78,b] .= 0.5 .* vv
        end
        for (r,(_,_,ii,jj)) in enumerate(pairlist)
            x2a[:,r] .= -(A_aa \ (A_ad*x2[:,r] + He[aidx,ii,jj]))
            qcross[(k-1)*nout+1:k*nout,r] .= Cfull*full_state(x2[:,r],x2a[:,r]) + Hobse[:,ii,jj]
        end
    end
end

CSV.write(joinpath(RES,"hessian_contraction_checks.csv"),symrows)
CSV.write(joinpath(RES,"hessian_subsystem_breakdown.csv"),subrows)
CSV.write(joinpath(RES,"algebraic_onset_checks.csv"),onset)
writedlm(joinpath(RES,"analytic_D_all.csv"),danalytic,',')
writedlm(joinpath(RES,"analytic_Q_self_all.csv"),qself,',')
writedlm(joinpath(RES,"analytic_Q_voltage_all.csv"),qvoltage,',')
writedlm(joinpath(RES,"analytic_Q_cross_all.csv"),qcross,',')

vr = randn(n+np); wr = randn(n+np)
mh = ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> obsfun(x0+s*vr[1:n]+t*wr[1:n],p0+s*vr[n+1:end]+t*wr[n+1:end]),0.0),0.0)
CSV.write(joinpath(RES,"measurement_second_order_audit.csv"),DataFrame(channel=1:length(mh),second_directional_value=mh,status=abs.(mh).<1e-10))
open(joinpath(RES,"v3_exclusion_manifest_delta.csv"),"w") do io
    println(io,"namespace,physical_trajectories,noise_seeds,status")
    println(io,"ANALYTIC_SECOND_ORDER_DAE_V1,0,0,NO_NEW_SIMULATIONS")
end
open(joinpath(RES,"subsystem_failure_localization.csv"),"w") do io
    println(io,"subsystem,status,reason")
    println(io,"algebraic_network,NOT_SEPARATELY_LOCALIZED,global_directional_contraction_export")
    println(io,"dynamic_model,NOT_SEPARATELY_LOCALIZED,global_directional_contraction_export")
    println(io,"controllers,NOT_SEPARATELY_LOCALIZED,global_directional_contraction_export")
    println(io,"load_model,NOT_SEPARATELY_LOCALIZED,global_directional_contraction_export")
    println(io,"event_parameterization,PASS,p(a)=p0*(1+a); p_second=0")
    println(io,"PMU_measurement,PASS,affine_ReIm_voltage_and_current_map")
end
println("analytic_second_order_dae_export_done n=",n," nd=",nd," na=",na," pairs=",npair)
