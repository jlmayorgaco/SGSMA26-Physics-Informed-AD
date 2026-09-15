# Analysis for EVENT-MAP-ALIGNMENT-PILOT-V1. Production maps are generated
# by pd39_event_map_alignment_flow_v1.jl; this file differentiates those exact
# maps and propagates only the frozen analytic variational equations after t1.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII, ForwardDiff
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random, DelimitedFiles, Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "event_map_alignment_pilot_v1")
const FLOW = joinpath(OUT, "flow_maps")
const BASE = joinpath(ROOT, "output", "second_order_op_robustness_v1")
const RES = joinpath(OUT, "results")
mkpath(RES)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const BUSES = [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
const OBS = [2,5,6,10,19,22,29,39]
const EDGE_FOR = [11,37,18,35,31,1,8,5]
const NATIVE_FOR = [false,false,true,true,true,false,false,false]
const DT = 1 / 30
const TAU = 2.0
const NF = 30

function build()
    mod = Module(Symbol("EVENT_MAP_ANALYSIS_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase,
        NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve))
    redirect_stdout(devnull) do
        Base.include_string(mod, read(SRC, String), "event_map_analysis_ieee39.jl")
    end
    nw = getfield(mod, :nw)
    Core.eval(mod, quote
        formula = @initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula)
        set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs = Base.invokelatest(solve_powerflow, nw; verbose=false)
    s0 = Base.invokelatest(initialize_from_pf!, nw; pfs=pfs, verbose=false)
    nw, s0
end

nw, s0 = build()
x0 = Float64.(uflat(s0)); p0 = Float64.(pflat(s0)); n = length(x0); np = length(p0)
M = Matrix(nw.mass_matrix); md = diag(M)
didx = findall(!iszero, md); aidx = findall(iszero, md); nd, na = length(didx), length(aidx)
syms = SII.variable_symbols(nw)
state_index(sym::String) = only(findall(s -> string(s) == sym, syms))
vri = [state_index("VIndex($(b), :busbar₊u_r)") for b in 1:39]
vii = [state_index("VIndex($(b), :busbar₊u_i)") for b in 1:39]

function pmu_coefficients()
    data = CSV.read(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data", "branch.csv"), DataFrame)
    out = Tuple{Int,Int,ComplexF64,ComplexF64,Bool}[]
    for (edge,native) in zip(EDGE_FOR,NATIVE_FOR)
        row=data[edge,:]; i=Int(row.src_bus); j=Int(row.dst_bus)
        ys=1/complex(Float64(row.R),Float64(row.X)); tap=Float64(row.r_src)==0 ? 1.0 : Float64(row.r_src)
        aa=-(ys+complex(Float64(row.G_src),Float64(row.B_src)))*tap^2; bb=ys*tap
        cc=ys*tap; dd=-(ys+complex(Float64(row.G_dst),Float64(row.B_dst)))
        push!(out,(i,j,native ? aa : cc,native ? bb : dd,native))
    end
    out
end
const PMU_COEFF = pmu_coefficients()
function pmu(x)
    T=eltype(x); y=zeros(T,32); r=1
    for b in OBS; y[r]=x[vri[b]]; y[r+1]=x[vii[b]]; r+=2; end
    for (i,j,a,b,_) in PMU_COEFF
        vr=a.re*x[vri[i]]-a.im*x[vii[i]]+b.re*x[vri[j]]-b.im*x[vii[j]]
        vi=a.re*x[vii[i]]+a.im*x[vri[i]]+b.re*x[vii[j]]+b.im*x[vri[j]]
        y[r]=vr; y[r+1]=vi; r+=2
    end
    y
end
# Use the frozen, independently validated PMU Jacobian for the propagation
# and TDS replay.  The affine PMU map is exact; loading this artifact also
# prevents a second implementation of the native-state ordering from entering
# the audit.  The AD Jacobian is retained as a parity check below.
const Cfull = Matrix{Float64}(readdlm(joinpath(BASE,"analytic_nominal","results","C_pmu_frozen.csv"),',',Float64))
const C_ad = ForwardDiff.jacobian(pmu, x0)
CSV.write(joinpath(RES,"pmu_map_alignment_audit.csv"), DataFrame(check=["frozen_vs_AD"],
    relative_error=[norm(Cfull-C_ad)/max(norm(Cfull),1e-300)], status=[norm(Cfull-C_ad)/max(norm(Cfull),1e-300)<1e-8 ? "PASS" : "FAIL_LOCAL_MAP_REVIEW"]))
residual(x,p) = begin
    du=zeros(eltype(x),n); nw(du,x,p,0.0); du
end

function load_case(label)
    d=CSV.read(joinpath(FLOW,label*".csv"),DataFrame)
    uc=sort([c for c in names(d) if startswith(String(c),"u")], by=x->parse(Int,replace(String(x),"u"=>"")))
    Float64.(d.time), Matrix{Float64}(d[:,uc])
end
function at_time(label,targ)
    t,u=load_case(label); hits=findall(abs.(t.-targ).<1e-9); k=isempty(hits) ? argmin(abs.(t.-targ)) : last(hits); u[k,:],t[k]
end
token(h) = replace(string(h),"."=>"p")
function state_derivatives(h, kind)
    tok=token(h); base,_=at_time("base_h"*tok,TAU+DT)
    if kind==:self
        pp,_=at_time("self7_p_h"*tok,TAU+DT); mm,_=at_time("self7_m_h"*tok,TAU+DT)
        return (pp-mm)/(2h),(pp-2base+mm)/(h*h)
    else
        pp,_=at_time("cross712_pp_h"*tok,TAU+DT); pm,_=at_time("cross712_pm_h"*tok,TAU+DT)
        mp,_=at_time("cross712_mp_h"*tok,TAU+DT); mm,_=at_time("cross712_mm_h"*tok,TAU+DT)
        return (pp+pm-mp-mm)/(4h),(pp-pm+mp-mm)/(4h),(pp-pm-mp+mm)/(4h*h)
    end
end
function richardson(kind, order)
    d25=state_derivatives(0.0025,kind)[order]; d5=state_derivatives(0.005,kind)[order]
    (4d25-d5)/3
end
function richardson_cross(which)
    order = which == 7 ? 1 : which == 12 ? 2 : 3
    d25=state_derivatives(0.0025,:cross)[order]; d5=state_derivatives(0.005,:cross)[order]
    (4d25-d5)/3
end

# Frozen nominal descriptor linearization and physical parameter directions.
desc = linearize_network(s0); A = Matrix(desc.A)
A_dd=A[didx,didx]; A_da=A[didx,aidx]; A_ad=A[aidx,didx]; A_aa=A[aidx,aidx]
As=A_dd-A_da*(A_aa\A_ad)
E_cache=Dict{Float64,Matrix{Float64}}(); G_cache=Dict{Float64,Matrix{Float64}}()
function EG(dt)
    if !haskey(E_cache,dt)
        z=zeros(Float64,2nd,2nd); z[1:nd,1:nd].=As; z[1:nd,nd+1:end].=Matrix(I,nd,nd)
        ez=exp(z*dt); E_cache[dt]=ez[1:nd,1:nd]; G_cache[dt]=ez[1:nd,nd+1:end]
    end
    E_cache[dt],G_cache[dt]
end
Pdirs=Dict{Int,Vector{Float64}}(); Bfulls=Dict{Int,Vector{Float64}}(); Bds=Dict{Int,Vector{Float64}}()
for bus in (7,12)
    psym=VIndex(bus,:ZIPLoad₊Pset); qsym=VIndex(bus,:ZIPLoad₊Qset)
    ip=only(SII.parameter_index(nw,psym)); iq=only(SII.parameter_index(nw,qsym))
    pd=zeros(Float64,np); pd[ip]=p0[ip]; pd[iq]=p0[iq]; Pdirs[bus]=pd
    lf=linearize_network(s0; in=[psym,qsym], out=syms); bf=Matrix(lf.B)*[p0[ip],p0[iq]]
    Bfulls[bus]=bf; Bds[bus]=bf[didx]-A_da*(A_aa\bf[aidx])
end
function complete_first(xd,bus)
    xa=-(A_aa\(A_ad*xd+Bfulls[bus][aidx])); xf=zeros(Float64,n); xf[didx].=xd; xf[aidx].=xa; xf
end
function first_advance(xd,bus,dt)
    E,G=EG(dt); E*xd+G*Bds[bus]
end
function Hcontract(x1,x2,bus1,bus2)
    # Exact residual directional HVP at the nominal equilibrium.
    Vx=hcat(x1,x2); Vp=hcat(Pdirs[bus1],Pdirs[bus2]); out=zeros(Float64,n)
    for k in 1:n
        fk(th)=residual(x0+Vx*th,p0+Vp*th)[k]
        out[k]=ForwardDiff.hessian(fk,zeros(Float64,2))[1,2]
    end
    out
end
function Hraw(x1,x2,bus1,bus2)
    Vx=hcat(x1,x2); Vp=hcat(Pdirs[bus1],Pdirs[bus2]); out=zeros(Float64,n)
    for k in 1:n
        fk(th)=residual(x0+Vx*th,p0+Vp*th)[k]
        out[k]=ForwardDiff.hessian(fk,zeros(Float64,2))[1,2]
    end
    out
end
function second_forcing(x1,x2,bus1,bus2)
    H=Hraw(x1,x2,bus1,bus2); H[didx]-A_da*(A_aa\H[aidx])
end
function second_advance(xd1,xd2,xd2state,bus1,bus2,dt)
    x1m=first_advance(xd1,bus1,dt/2); x2m=first_advance(xd2,bus2,dt/2)
    Hm=second_forcing(complete_first(x1m,bus1),complete_first(x2m,bus2),bus1,bus2)
    E,G=EG(dt); E*xd2state+G*Hm
end

function self_existing_onset(bus)
    xd=first_advance(zeros(Float64,nd),bus,DT)
    xm=first_advance(zeros(Float64,nd),bus,DT/2)
    H=second_forcing(complete_first(xm,bus),complete_first(xm,bus),bus,bus)
    _,G=EG(DT); xd,G*H
end
function cross_existing_onset()
    x7=first_advance(zeros(Float64,nd),7,DT); x12=first_advance(zeros(Float64,nd),12,DT)
    m7=first_advance(zeros(Float64,nd),7,DT/2); m12=first_advance(zeros(Float64,nd),12,DT/2)
    H=second_forcing(complete_first(m7,7),complete_first(m12,12),7,12); _,G=EG(DT)
    x7,x12,G*H
end

# Persist first-step derivatives and Richardson uncertainty.
pr=DataFrame(direction=String[],h=Float64[],derivative_order=Int[],coordinate=Int[],value=Float64[],time_s=Float64[])
unc=DataFrame(direction=String[],derivative_order=Int[],norm_h0025=Float64[],norm_h005=Float64[],norm_h01=Float64[],richardson_error_h0025=Float64[],richardson_error_h01=Float64[])
for (kind,name) in ((:self,"self_bus7"),(:cross,"cross_bus7_12"))
    ords = kind == :self ? (1,2) : (1,2,3)
    for ord in ords
        a=state_derivatives(0.0025,kind)[ord]; b=state_derivatives(0.005,kind)[ord]; c=state_derivatives(0.01,kind)[ord]; r=(4a-b)/3
        for k in 1:n; push!(pr,(name,0.0025,ord,k,a[k],TAU+DT)); end
        push!(unc,(name,ord,norm(a),norm(b),norm(c),norm(r-a),norm(r-c)))
    end
end
CSV.write(joinpath(RES,"production_first_step_derivatives.csv"),pr)
CSV.write(joinpath(RES,"numerical_uncertainty.csv"),unc)

# Physical PMU Q Richardson replay from the exact production maps.
function q_tds_self(h)
    tok=token(h); y0=Vector{Vector{Float64}}(); yp=Vector{Vector{Float64}}(); ym=Vector{Vector{Float64}}()
    for k in 1:NF
        targ=TAU+k*DT
        push!(y0,pmu(at_time("base_h"*tok,targ)[1])); push!(yp,pmu(at_time("self7_p_h"*tok,targ)[1])); push!(ym,pmu(at_time("self7_m_h"*tok,targ)[1]))
    end
    reduce(vcat,[reshape(0.5*(yp[k]-2y0[k]+ym[k])/(h*h),1,:) for k in 1:NF])
end
function q_tds_cross(h)
    tok=token(h); labs=["cross712_pp_h"*tok,"cross712_pm_h"*tok,"cross712_mp_h"*tok,"cross712_mm_h"*tok]
    y=Dict(l=>Vector{Vector{Float64}}() for l in labs)
    for k in 1:NF, l in labs; push!(y[l],pmu(at_time(l,TAU+k*DT)[1])); end
    reduce(vcat,[reshape((y[labs[1]][k]-y[labs[2]][k]-y[labs[3]][k]+y[labs[4]][k])/(4h*h),1,:) for k in 1:NF])
end
qtdsS=(4q_tds_self(0.0025)-q_tds_self(0.005))/3
qtdsX=(4q_tds_cross(0.0025)-q_tds_cross(0.005))/3

function full_second(x2d,x1d1,x1d2,bus1,bus2)
    H=Hraw(complete_first(x1d1,bus1),complete_first(x1d2,bus2),bus1,bus2)
    x2a=-(A_aa\(A_ad*x2d+H[aidx])); xf=zeros(Float64,n); xf[didx].=x2d; xf[aidx].=x2a; xf
end
function aligned_curve(kind)
    d1=kind==:self ? richardson(kind,1) : richardson_cross(7); d12=kind==:self ? d1 : richardson_cross(12); d2=kind==:self ? richardson(kind,2) : richardson_cross(3)
    b1=7; b2=kind==:self ? 7 : 12; x1=d1[didx]; x12=d12[didx]; x2=d2[didx]; out=zeros(Float64,NF,32)
    # First saved post-event sample is the exact production-map derivative.
    out[1,:].=kind==:self ? 0.5.*(Cfull*d2) : Cfull*d2
    for k in 2:NF
        x2=second_advance(x1,x12,x2,b1,b2,DT); x1=first_advance(x1,b1,DT); x12=kind==:self ? x1 : first_advance(x12,b2,DT)
        xf=full_second(x2,x1,x12,b1,b2)
        out[k,:].=kind==:self ? 0.5.*(Cfull*xf) : Cfull*xf
    end
    out
end
function existing_curve(kind)
    if kind==:self; x1,x2=self_existing_onset(7); b1=7; b2=7
    else; x1,x12,x2=cross_existing_onset(); b1=7; b2=12; end
    out=zeros(Float64,NF,32)
    out[1,:].=kind==:self ? 0.5.*(Cfull*full_second(x2,x1,x1,b1,b2)) : Cfull*full_second(x2,x1,x12,b1,b2)
    for k in 2:NF
        x2=second_advance(x1,kind==:self ? x1 : x12,x2,b1,b2,DT)
        x1=first_advance(x1,b1,DT); x12=kind==:self ? x1 : first_advance(x12,12,DT)
        xf=full_second(x2,x1,x12,b1,b2)
        out[k,:].=kind==:self ? 0.5.*(Cfull*xf) : Cfull*xf
    end
    out
end
alignedS=aligned_curve(:self); alignedX=aligned_curve(:cross)
existingS=existing_curve(:self); existingX=existing_curve(:cross)

rows=DataFrame(direction=String[],frame=Int[],tau_s=Float64[],channel=Int[],aligned=Float64[],existing=Float64[],tds_richardson=Float64[])
for (name,a,e,t) in (("self_bus7",alignedS,existingS,qtdsS),("cross_bus7_12",alignedX,existingX,qtdsX))
    for k in 1:NF, c in 1:32
        push!(rows,(name,k,k*DT,c,a[k,c],e[k,c],t[k,c]))
    end
end
CSV.write(joinpath(RES,"aligned_vs_existing_vs_tds.csv"),rows)

hom=DataFrame(direction=String[],frame=Int[],tau_s=Float64[],observed_norm=Float64[],predicted_norm=Float64[],relative_error=Float64[],cosine=Float64[])
cosv(a,b)=dot(a,b)/max(norm(a)*norm(b),1e-300)
function complete_second_homogeneous(xd)
    xa=-(A_aa\(A_ad*xd)); xf=zeros(Float64,n); xf[didx].=xd; xf[aidx].=xa; xf
end
function homogeneous(kind)
    if kind==:self; d2=richardson(:self,2); _,e2=self_existing_onset(7); b=7; nm="self_bus7"
    else; d2=richardson_cross(3); _,_,e2=cross_existing_onset(); b=7; nm="cross_bus7_12"; end
    # observed = existing - aligned, hence the initial homogeneous state
    # mismatch has the same orientation: existing onset minus production map.
    de=e2-d2[didx]; E,_=EG(DT); aa=kind==:self ? existingS : existingX; al=kind==:self ? alignedS : alignedX
    for k in 1:NF
        pred=E^(k-1)*de; obs=norm(aa[k,:]-al[k,:]); pp=(kind==:self ? 0.5 : 1.0).*(Cfull*complete_second_homogeneous(pred)); pn=norm(pp)
        push!(hom,(nm,k,k*DT,obs,pn,abs(obs-pn)/max(obs,1e-300),cosv(aa[k,:]-al[k,:],pp)))
    end
end
homogeneous(:self); homogeneous(:cross); CSV.write(joinpath(RES,"homogeneous_error_dynamics.csv"),hom)
println("event_map_alignment_analysis_done rows=",nrow(rows)," output=",OUT)
