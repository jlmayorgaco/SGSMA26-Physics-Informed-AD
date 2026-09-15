# FINITE-AMPLITUDE-REMAINDER-ORDER-V1 production TDS executor.
# It consumes the frozen Python point manifest; no points are inferred here.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, SHA, Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "finite_amplitude_remainder_order_v1")
const RES = joinpath(OUT, "results")
const PHYS = joinpath(OUT, "physical")
const POINTS = get(ENV, "FINITE_AMP_POINTS_FILE", joinpath(RES, "amplitude_homotopy_manifest.csv"))
mkpath(PHYS); mkpath(RES)
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")

function build(data::String)
    txt = replace(read(SRC,String), r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)" => "DATA_DIR = raw\"$data\"")
    mod = Module(Symbol("FINITE_AMP_", rand(UInt)))
    Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics,
                      OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CSV, DataFrames))
    redirect_stdout(devnull) do; Base.include_string(mod,txt,"finite_amp_part1.jl"); end
    nw=getfield(mod,:nw)
    Core.eval(mod, quote
        formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
        set_initformula!($nw[VIndex(31)], formula); set_initformula!($nw[VIndex(39)], formula)
    end)
    pfs=Base.invokelatest(solve_powerflow,nw;verbose=false)
    s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false)
    return nw,s0
end

function make_callbacks!(nw, supports::Vector{Int})
    refs=Dict{Int,Ref{Float64}}()
    for bus in supports
        ref=Ref(0.0); refs[bus]=ref; init=Ref(false); bp=Ref(0.0); bq=Ref(0.0)
        syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
        aff=ComponentAffect([],syms) do u,p,ctx
            if !init[]; bp[]=float(p[syms[1]]); bq[]=float(p[syms[2]]); init[]=true; end
            p[syms[1]]=bp[]*(1+ref[]); p[syms[2]]=bq[]*(1+ref[])
        end
        set_callback!(nw[VIndex(bus)],PresetTimeComponentCallback([2.0],aff))
    end
    return refs
end

function run_one(nw0,s00,r,dir)
    sp=split(String(r.support),"-"); buses=[parse(Int,x) for x in sp]
    amps=[Float64(r.amplitude_i),Float64(r.amplitude_j)]
    nw=deepcopy(nw0); s0=deepcopy(s00); refs=make_callbacks!(nw,buses)
    for (b,a) in zip(buses,amps); refs[b][]=a; end
    tid=String(r.point_id); path=joinpath(dir,tid*".csv")
    if isfile(path); return (tid,true,0.0,path,"CHECKPOINT"); end
    t0=time(); ok=false; ret="ERROR"
    try
        abst=String(r.precision)=="tight" ? 1e-11 : 1e-9
        sol=solve(ODEProblem(nw,s0,(0.0,6.1)),Rodas5P();abstol=abst,reltol=abst,saveat=1/30,dtmax=1/60,maxiters=10^7)
        ret=string(sol.retcode); ok=ret=="Success" && all(isfinite,sol.u[end])
        rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
        for t in sol.t,b in 1:39
            z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i))))
            push!(rows,(t,b,real(z),imag(z)))
        end
        ok && CSV.write(path,rows)
    catch err; ret=sprint(showerror,err); ok=false; end
    dt=time()-t0
    return (tid,ok,dt,path,ret)
end

function main()
    pts=CSV.read(POINTS,DataFrame); pts=pts[pts.new_tds .== true,:]
    manifest=joinpath(RES,"tds_execution_manifest.csv")
    out=isfile(manifest) ? CSV.read(manifest,DataFrame) : DataFrame(point_id=String[],op_tag=String[],status=String[],path=String[],runtime_s=Float64[],retcode=String[],sha256=String[],precision=String[])
    for (op,m) in (("op_m035",.35),("op_m085",.85),("op_m125",1.25))
        pp=pts[pts.op_tag .== op,:]; nrow(pp)==0 && continue
        data=joinpath(ROOT,"output","t120_multi_op_independent_validation_v1","op_data",op); nw,s0=build(data)
        dir=joinpath(PHYS,op,"results"); mkpath(dir)
        for r in eachrow(pp)
            x=run_one(nw,s0,r,dir); h=isfile(x[4]) ? bytes2hex(sha256(read(x[4]))) : ""
            if !(x[1] in out.point_id)
                push!(out,(x[1],op,x[2] ? (x[5]=="CHECKPOINT" ? "CHECKPOINT" : "EXECUTED_SUCCESS") : "EXECUTED_FAIL",x[4],x[3],x[5],h,String(r.precision)))
            end
            CSV.write(manifest,out); println(op," ",x[1]," ",x[2]," ",x[3]," ",x[5]); flush(stdout)
        end
        GC.gc()
    end
    CSV.write(manifest,out)
end
main()
