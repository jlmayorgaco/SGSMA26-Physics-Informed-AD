using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output",get(ENV,"ATLAS_OUT","load_response_atlas_v1")); const RES=joinpath(OUT,"results"); const CK=joinpath(OUT,"checkpoints"); mkpath(RES); mkpath(CK)
const SRC=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39_part1.jl"); const DATASRC=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39data")
const EPS=[0.0025,-0.0025,0.005,-0.005,0.01,-0.01,0.02,-0.02,0.10]
function make_data(); d=mktempdir(OUT); for f in readdir(DATASRC); CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame)); end; d end
function build(data)
 txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
 mod=Module(Symbol("ATLAS_",rand(UInt))); Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
 redirect_stdout(devnull) do
  Base.include_string(mod,txt,"atlas_part1.jl")
 end
 nw=getfield(mod,:nw)
 Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula) end)
 pfs=Base.invokelatest(solve_powerflow,nw;verbose=false); s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false); return nw,s0
end
function simulate(bus::Int, amp::Float64, rep::Int, data)
 cid="LOAD_BUS_$(bus)_A$(replace(string(amp),"-"=>"m","."=>"p"))_R$(rep)"; path=joinpath(RES,cid*".csv"); isfile(path) && return (cid,true,0.0)
 nw,s0=build(data)
 if amp != 0.0
  syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
  aff=ComponentAffect([],syms) do u,p,ctx; p[syms[1]] *= (1+amp); p[syms[2]] *= (1+amp); end
  set_callback!(nw[VIndex(bus)],PresetTimeComponentCallback([2.0],aff))
 end
 t0=time(); sol=solve(ODEProblem(nw,s0,(0.0,5.0)),Rodas5P();abstol=1e-9,reltol=1e-9,saveat=1/30); dt=time()-t0
 rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
 for t in sol.t, b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z))); end
 CSV.write(path,rows); open(joinpath(CK,cid*".done"),"w") do io; write(io,"retcode=$(sol.retcode),frames=$(length(sol.t)),seconds=$(dt)\n") end; return (cid,string(sol.retcode)=="Success",dt)
end
function main()
 load=CSV.read(joinpath(DATASRC,"load.csv"),DataFrame); bus=CSV.read(joinpath(DATASRC,"bus.csv"),DataFrame); valid=innerjoin(load,bus[!,[:bus,:bus_type,:has_load]],on=:bus); valid=valid[(valid.has_load .== true) .& (valid.bus_type .== "PQ") .& (.!(in.(valid.bus, Ref([2,5,6,10,19,22,29,39])))),:]
 requested=parse.(Int,split(get(ENV,"ATLAS_CANDIDATES","7,12"),",")); requested=intersect(requested,Int.(valid.bus)); reps=parse(Int,get(ENV,"ATLAS_REPS","1"));
 mf=DataFrame(candidate_bus=Int[],amplitude=Float64[],realization=Int[],case_id=String[],status=String[],path=String[])
 amps=[0.0; parse.(Float64,split(get(ENV,"ATLAS_AMPS",join(string.(EPS),",")),","))]
 for b in requested, rep in 1:reps, amp in amps
  cid,ok,dt=simulate(b,amp,rep,make_data()); push!(mf,(b,amp,rep,cid,ok ? "EXECUTED_SUCCESS" : "EXECUTED_FAIL",joinpath(RES,cid*".csv"))); println(cid," ",ok," ",dt)
 end
 CSV.write(joinpath(OUT,"simulation_manifest_native.csv"),mf); CSV.write(joinpath(OUT,"candidate_registry_native.csv"),valid)
end
main()
