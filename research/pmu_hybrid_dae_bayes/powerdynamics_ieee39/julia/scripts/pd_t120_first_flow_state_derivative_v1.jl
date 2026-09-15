# Full 192-state first-flow maps for T120-FIRST-FLOW-STATE-DERIVATIVE-V1.
# Audit-only: reuses the three already excluded M6 operating points and the
# native production callback.  No estimator or prospective V3 data is touched.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using NetworkDynamics: SII
using CSV, DataFrames, LinearAlgebra, Random, SHA, DelimitedFiles

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "t120_first_flow_state_derivative_v1")
const MAP = joinpath(OUT, "state_maps"); const META = joinpath(OUT, "metadata")
mkpath(MAP); mkpath(META)
const PKG = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU = 2.0; const DT = 1/30; const T1 = TAU + DT; const HORIZON = 5.0
const TOL = 1e-10
const SELF_BUSES = [7,26,3,16]
const CROSS_PAIRS = [(7,12),(26,28),(3,18),(16,18)]
const OPS = [("op_m035",0.35),("op_m085",0.85),("op_m125",1.25)]

function copy_data(tag,m)
    d = mktempdir(OUT)
    for f in readdir(PKG); CSV.write(joinpath(d,f), CSV.read(joinpath(PKG,f),DataFrame)); end
    bus = CSV.read(joinpath(d,"bus.csv"),DataFrame); k=only(findall(Int.(bus.bus).==3))
    bus[k,:P] = Float64(bus[k,:P])*(1+0.03*m); bus[k,:Q] = Float64(bus[k,:Q])*(1+0.03*m)
    CSV.write(joinpath(d,"bus.csv"),bus); return d
end

function build(data)
    txt=replace(read(SRC,String), r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" => "DATA_DIR = raw\"$data\"")
    mod=Module(Symbol("FIRST_FLOW_",rand(UInt)))
    Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
    redirect_stdout(devnull) do
        Base.include_string(mod,txt,"first_flow_ieee39.jl")
    end
    nw=getfield(mod,:nw)
    Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula) end)
    pf=Base.invokelatest(solve_powerflow,nw;verbose=false); s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pf,verbose=false)
    return nw,s0
end

function run_op(tag,m)
    data=copy_data(tag,m); nw,s0=build(data); n=length(uflat(s0)); pms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
    refs=Dict{Int,Ref{Float64}}(); nominal=Dict{Int,Tuple{Float64,Float64}}()
    buses=unique(vcat(SELF_BUSES,[x for pr in CROSS_PAIRS for x in pr]))
    for b in buses
        ref=Ref(0.0); refs[b]=ref
        aff=ComponentAffect([],pms) do u,p,ctx
            if !haskey(nominal,b); nominal[b]=(Float64(p[pms[1]]),Float64(p[pms[2]])); end
            p[pms[1]]=nominal[b][1]*(1+ref[]); p[pms[2]]=nominal[b][2]*(1+ref[])
        end
        set_callback!(nw[VIndex(b)],PresetTimeComponentCallback([TAU],aff))
    end
    od=joinpath(MAP,tag); mkpath(od); rows=DataFrame(case=String[],op_tag=String[],op_m=Float64[],amp_i=Float64[],amp_j=Float64[],time_s=Float64[],retcode=String[],runtime_s=Float64[])
    states=DataFrame(case=String[],op_tag=String[],op_m=Float64[],amp_i=Float64[],amp_j=Float64[],time_s=Float64[])
    # Baseline at the same pre-event operating point is required for centered
    # second derivatives; the callback fires with zero parameter change.
    bpath=joinpath(od,"baseline.csv")
    if !isfile(bpath)
        for b in keys(refs); refs[b][]=0.0; end
        t0=time(); sol=solve(ODEProblem(nw,deepcopy(s0),(0.0,HORIZON)),Rodas5P();abstol=TOL,reltol=TOL,dtmax=1/60,maxiters=10^7,saveat=DT); dt=time()-t0
        u=Float64.(sol(T1)); df=DataFrame(reshape(u,1,:),:auto); rename!(df,Symbol.("u" .* string.(1:n))); insertcols!(df,1,:time_s=>T1); insertcols!(df,1,:amp_j=>0.0); insertcols!(df,1,:amp_i=>0.0); insertcols!(df,1,:op_m=>m); insertcols!(df,1,:op_tag=>tag); insertcols!(df,1,:case=>"baseline"); CSV.write(bpath,df); push!(rows,("baseline",tag,m,0.0,0.0,T1,string(sol.retcode),dt)); push!(states,("baseline",tag,m,0.0,0.0,T1)); println(tag," baseline ",string(sol.retcode)," ",dt)
    end
    for (kind,bi,bj) in vcat([("self",b,0) for b in SELF_BUSES],[("cross",p[1],p[2]) for p in CROSS_PAIRS])
        hs = kind=="self" ? [(0.005,0.0),(-0.005,0.0),(0.0025,0.0),(-0.0025,0.0)] : [(0.005,0.005),(0.005,-0.005),(-0.005,0.005),(-0.005,-0.005),(0.0025,0.0025),(0.0025,-0.0025),(-0.0025,0.0025),(-0.0025,-0.0025)]
        for (ai,aj) in hs
            cid=kind*"_$(bi)"*(bj==0 ? "" : "_$(bj)")*"_ai$(replace(string(ai),"-"=>"m","."=>"p"))_aj$(replace(string(aj),"-"=>"m","."=>"p"))"
            if isfile(joinpath(od,cid*".csv")); continue; end
            for b in keys(refs); refs[b][]=0.0; end; refs[bi][]=ai; bj>0 && (refs[bj][]=aj)
            t0=time(); sol=solve(ODEProblem(nw,deepcopy(s0),(0.0,HORIZON)),Rodas5P();abstol=TOL,reltol=TOL,dtmax=1/60,maxiters=10^7,saveat=DT); dt=time()-t0
            ret=string(sol.retcode); u=Float64.(sol(T1)); length(u)==n || error("state length mismatch")
            df=DataFrame(reshape(u,1,:),:auto); rename!(df,Symbol.("u" .* string.(1:n))); insertcols!(df,1,:time_s=>T1); insertcols!(df,1,:amp_j=>aj); insertcols!(df,1,:amp_i=>ai); insertcols!(df,1,:op_m=>m); insertcols!(df,1,:op_tag=>tag); insertcols!(df,1,:case=>cid); CSV.write(joinpath(od,cid*".csv"),df)
            push!(rows,(cid,tag,m,ai,aj,T1,ret,dt)); push!(states,(cid,tag,m,ai,aj,T1)); println(tag," ",cid," ",ret," ",dt); GC.gc()
        end
    end
    CSV.write(joinpath(od,"manifest.csv"),rows); CSV.write(joinpath(od,"state_manifest.csv"),states)
    syms=SII.variable_symbols(nw); M=Matrix(nw.mass_matrix); CSV.write(joinpath(META,"state_order_$(tag).csv"),DataFrame(index=1:n,symbol=string.(syms),mass=diag(M),kind=[iszero(x) ? "algebraic" : "differential" for x in diag(M)])); writedlm(joinpath(META,"mass_matrix_$(tag).csv"),M,',')
end

for (tag,m) in OPS; run_op(tag,m); end
println("first_flow_state_derivative_export_done")
