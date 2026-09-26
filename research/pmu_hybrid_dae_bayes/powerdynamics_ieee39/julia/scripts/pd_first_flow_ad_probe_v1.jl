# Feasibility probe only.  It attempts ForwardDiff through the exact sampled
# production callback/solve map without changing simulator semantics.  A
# failure is recorded as evidence that Richardson must remain the canonical
# discrete-event-map reference.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using NetworkDynamics: ForwardDiff
using CSV, DataFrames, Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "first_flow_hessian_closure_v1")
const RES = joinpath(OUT, "results"); mkpath(RES)
const PKG = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39data")
const SRC = joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part1.jl")
const TAU=2.0; const T1=TAU+1/30; const TOL=1e-10

function build()
    mod=Module(Symbol("AD_PROBE_",rand(UInt)))
    Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve))
    redirect_stdout(devnull) do
        Base.include_string(mod,read(SRC,String),"first_flow_ad_probe_ieee39.jl")
    end
    nw=getfield(mod,:nw)
    Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula) end)
    pf=Base.invokelatest(solve_powerflow,nw;verbose=false)
    s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pf,verbose=false)
    nw,s0
end

nw,s0=build(); syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
amp=Ref(0.0); nominal=Ref{Union{Nothing,Tuple{Float64,Float64}}}(nothing)
aff=ComponentAffect([],syms) do u,p,ctx
    if nominal[] === nothing; nominal[]=(Float64(p[syms[1]]),Float64(p[syms[2]])); end
    p[syms[1]]=nominal[][1]*(1+amp[]); p[syms[2]]=nominal[][2]*(1+amp[])
end
set_callback!(nw[VIndex(7)],PresetTimeComponentCallback([TAU],aff))
flow(a) = begin
    amp[] = a
    sol=solve(ODEProblem(nw,deepcopy(s0),(0.0,T1)),Rodas5P();abstol=TOL,reltol=TOL,dtmax=1/60,maxiters=10^7,saveat=[T1])
    Float64(sol(T1)[1])
end
rows=DataFrame(method=String[],status=String[],detail=String[])
for (name,fun) in [("ForwardDiff_through_callback_solve",()->ForwardDiff.derivative(flow,0.0))]
    try
        val=fun(); push!(rows,(name,"PASS",string(val)))
    catch err
        push!(rows,(name,"NOT_SUPPORTED",sprint(showerror,err,catch_backtrace())))
    end
end
CSV.write(joinpath(RES,"ad_feasibility.csv"),rows)
println("first_flow_ad_probe_done status=",rows.status[1])
