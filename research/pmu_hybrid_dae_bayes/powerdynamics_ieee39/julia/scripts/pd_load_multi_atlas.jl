using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, Random

"""Native true-time-local two-load event atlas.

Each trajectory starts from the same nominal initialized state. Two separate
PresetTimeComponentCallbacks fire at t=2 s, so the pre-event state is not
reinitialized. The output contract is identical to the single-load atlas.
"""
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output",get(ENV,"MULTI_OUT","load_multi_bayes_v1")); const RES=joinpath(OUT,"physical","results"); const CK=joinpath(OUT,"physical","checkpoints"); mkpath(RES); mkpath(CK)
const SRC=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39_part1.jl"); const DATASRC=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39data")

function make_data(); d=mktempdir(OUT); for f in readdir(DATASRC); CSV.write(joinpath(d,f),CSV.read(joinpath(DATASRC,f),DataFrame)); end; d end
function build(data)
 txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\"")
 mod=Module(Symbol("MULTI_",rand(UInt))); Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames))
 redirect_stdout(devnull) do
  Base.include_string(mod,txt,"multi_part1.jl")
 end
 nw=getfield(mod,:nw)
 Core.eval(mod,quote formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2); set_initformula!($nw[VIndex(31)],formula); set_initformula!($nw[VIndex(39)],formula) end)
 pfs=Base.invokelatest(solve_powerflow,nw;verbose=false); s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pfs,verbose=false); return nw,s0
end

function tag(a::Float64); replace(string(a),"-"=>"m","."=>"p") end
function simulate(bi::Int,bj::Int,ai::Float64,aj::Float64,rep::Int,data)
 cid="PAIR_$(bi)_$(bj)_AI$(tag(ai))_AJ$(tag(aj))_R$(rep)"; path=joinpath(RES,cid*".csv"); isfile(path) && return (cid,true,0.0)
 nw,s0=build(data)
 events = bj == 0 ? ((bi,ai),) : ((bi,ai),(bj,aj))
 for (bus,amp) in events
  syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
  aff=ComponentAffect([],syms) do u,p,ctx; p[syms[1]] *= (1+amp); p[syms[2]] *= (1+amp); end
  set_callback!(nw[VIndex(bus)],PresetTimeComponentCallback([2.0],aff))
 end
 t0=time(); sol=solve(ODEProblem(nw,s0,(0.0,5.0)),Rodas5P();abstol=1e-9,reltol=1e-9,saveat=1/30); dt=time()-t0
 rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
 for t in sol.t, b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z))); end
 CSV.write(path,rows); open(joinpath(CK,cid*".done"),"w") do io; write(io,"retcode=$(sol.retcode),frames=$(length(sol.t)),seconds=$(dt)\n") end; return (cid,string(sol.retcode)=="Success",dt)
end

# Qij runs four signed combinations for a pair.  Building the compiled
# NetworkDynamics graph dominates runtime; clone one template per pair and
# run each sign from an independent copy.  This preserves the exact
# true-time-local callback semantics while avoiding four recompilations.
function simulate_pair_qij(bi::Int,bj::Int,h::Float64,rep::Int,data)
 nw_template,s0_template=build(data)
 for (ai,aj) in ((h,h),(h,-h),(-h,h),(-h,-h))
  cid="PAIR_$(bi)_$(bj)_AI$(tag(ai))_AJ$(tag(aj))_R$(rep)"; path=joinpath(RES,cid*".csv")
  isfile(path) && continue
  nw=deepcopy(nw_template); s0=deepcopy(s0_template)
  for (bus,amp) in ((bi,ai),(bj,aj))
   syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
   aff=ComponentAffect([],syms) do u,p,ctx; p[syms[1]] *= (1+amp); p[syms[2]] *= (1+amp); end
   set_callback!(nw[VIndex(bus)],PresetTimeComponentCallback([2.0],aff))
  end
  t0=time(); sol=solve(ODEProblem(nw,s0,(0.0,5.0)),Rodas5P();abstol=1e-9,reltol=1e-9,saveat=1/30); dt=time()-t0
  rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
  for t in sol.t, b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z))); end
  CSV.write(path,rows); open(joinpath(CK,cid*".done"),"w") do io; write(io,"retcode=$(sol.retcode),frames=$(length(sol.t)),seconds=$(dt)\n") end
  println(cid," ",string(sol.retcode)=="Success"," ",dt); GC.gc()
 end
end

# Memory-safe batch path: one compiled graph per pair, with callback closures
# whose amplitudes are updated between solves.  The callback stores nominal
# P/Q once and writes absolute values, so each run starts from the same
# pre-event plant without rebuilding or accumulating parameter changes.
function simulate_pair_reuse(bi::Int,bj::Int,cases_pair,data)
 # Avoid recompiling pairs that were already checkpointed by an earlier
 # interrupted process.
 if all(isfile(joinpath(RES,"PAIR_$(bi)_$(bj)_AI$(tag(ai))_AJ$(tag(aj))_R$(rep).csv")) for (ai,aj,rep) in cases_pair)
  return
 end
 nw,s0=build(data); refs=Dict{Int,Any}();
 for bus in (bi,bj)
  ref=Ref(0.0); refs[bus]=ref; init=Ref(false); bp=Ref(0.0); bq=Ref(0.0)
  syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
  aff=ComponentAffect([],syms) do u,p,ctx
   if !init[]; bp[]=float(p[syms[1]]); bq[]=float(p[syms[2]]); init[]=true; end
   p[syms[1]]=bp[]*(1+ref[]); p[syms[2]]=bq[]*(1+ref[])
  end
  set_callback!(nw[VIndex(bus)],PresetTimeComponentCallback([2.0],aff))
 end
 for (ai,aj,rep) in cases_pair
  cid="PAIR_$(bi)_$(bj)_AI$(tag(ai))_AJ$(tag(aj))_R$(rep)"; path=joinpath(RES,cid*".csv"); isfile(path) && continue
  refs[bi][]=ai; refs[bj][]=aj
  t0=time(); sol=solve(ODEProblem(nw,deepcopy(s0),(0.0,5.0)),Rodas5P();abstol=1e-9,reltol=1e-9,saveat=1/30); dt=time()-t0
  rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
  for t in sol.t, b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z))); end
  CSV.write(path,rows); open(joinpath(CK,cid*".done"),"w") do io; write(io,"retcode=$(sol.retcode),frames=$(length(sol.t)),seconds=$(dt)\n") end
  println(cid," ",string(sol.retcode)=="Success"," ",dt); GC.gc()
 end
end

function valid_buses()
 load=CSV.read(joinpath(DATASRC,"load.csv"),DataFrame); bus=CSV.read(joinpath(DATASRC,"bus.csv"),DataFrame); valid=innerjoin(load,bus[!,[:bus,:bus_type,:has_load]],on=:bus); Int.(valid[(valid.has_load .== true) .& (valid.bus_type .== "PQ") .& (.!(in.(valid.bus, Ref([2,5,6,10,19,22,29,39])))),:bus])
end
function main()
 valid=valid_buses(); requested=intersect(parse.(Int,split(get(ENV,"MULTI_CANDIDATES",join(string.(valid),",")),",")),valid); reps=parse(Int,get(ENV,"MULTI_REPS","1")); mode=get(ENV,"MULTI_MODE","QIJ")
 mf=DataFrame(source_i=Int[],source_j=Int[],amplitude_i=Float64[],amplitude_j=Float64[],realization=Int[],case_id=String[],status=String[],path=String[])
 cases=Tuple{Int,Int,Float64,Float64}[]
 pair_specs=Tuple{Int,Int}[]
 if haskey(ENV,"MULTI_PAIRS") && !isempty(ENV["MULTI_PAIRS"])
  for tok in split(ENV["MULTI_PAIRS"],";")
   ab=parse.(Int,split(tok,"-")); length(ab)==2 || error("invalid MULTI_PAIRS token: $tok")
   a,b=sort(ab); (a in requested && b in requested && a != b) && push!(pair_specs,(a,b))
  end
 else
  for ii in 1:length(requested)-1, jj in ii+1:length(requested); push!(pair_specs,(requested[ii],requested[jj])); end
 end
 if mode=="QIJ"
  h=parse(Float64,get(ENV,"MULTI_H","0.005")); for (bi,bj) in pair_specs; for (ai,aj) in ((h,h),(h,-h),(-h,h),(-h,-h)); push!(cases,(bi,bj,ai,aj)); end end
 elseif mode=="DEV"
  mags=[parse.(Float64,split(tok,",")) for tok in split(get(ENV,"MULTI_MAG_PAIRS","0.0001,0.0002;0.0005,0.001;0.002,0.004;0.01,0.025"),";")];
  for (bi,bj) in pair_specs; for (mi,mj) in mags, si in (-1.0,1.0), sj in (-1.0,1.0); push!(cases,(bi,bj,si*mi,sj*mj)); end end
 elseif mode=="SINGLE"
  amps=parse.(Float64,split(get(ENV,"MULTI_AMPS","0.0003,0.0008,0.003,0.007"),","));
  for bi in requested, a in amps, s in (-1.0,1.0); push!(cases,(bi,0,s*a,0.0)); end
 end
 # Reuse one immutable data cache across trajectories.  Building a fresh
 # temporary ieee39 data directory for every case leaks compiled state in
 # long runs and prevents reliable checkpoint/resume.
 data=make_data()
 manifest_path=joinpath(OUT,"simulation_manifest_native.csv")
 if get(ENV,"MULTI_REUSE","0")=="1" && mode!="SINGLE"
  for (bi,bj) in pair_specs
   cp=[(ai,aj,rep) for (bi2,bj2,ai,aj) in cases for rep in 1:reps if bi2==bi && bj2==bj]
   simulate_pair_reuse(bi,bj,cp,data)
   for (ai,aj,rep) in cp
    cid="PAIR_$(bi)_$(bj)_AI$(tag(ai))_AJ$(tag(aj))_R$(rep)"; path=joinpath(RES,cid*".csv")
    push!(mf,(bi,bj,ai,aj,rep,cid,isfile(path) ? "EXECUTED_SUCCESS" : "EXECUTED_FAIL",path))
   end
   CSV.write(manifest_path,mf); GC.gc()
  end
 else
  for (bi,bj,ai,aj) in cases, rep in 1:reps
   cid,ok,dt=simulate(bi,bj,ai,aj,rep,data)
   push!(mf,(bi,bj,ai,aj,rep,cid,ok ? "EXECUTED_SUCCESS" : "EXECUTED_FAIL",joinpath(RES,cid*".csv")))
   CSV.write(manifest_path,mf); println(cid," ",ok," ",dt); GC.gc()
  end
 end
 CSV.write(manifest_path,mf)
end
main()
