# SAMPLEWISE-SECOND-VARIATION-CLOSURE-V1 production TDS stencil executor.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using CSV, DataFrames, LinearAlgebra, SHA, Random
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output","samplewise_second_variation_closure_v1"); const RES=joinpath(OUT,"results"); const PHYS=joinpath(OUT,"physical")
const POINTS=get(ENV,"SAMPLEWISE_POINTS_FILE",joinpath(RES,"samplewise_manifest.csv")); mkpath(PHYS); mkpath(RES)
const SRC=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39_part1.jl")
function build(data)
 txt=replace(read(SRC,String),r"DATA_DIR = joinpath\(pkgdir\(PowerDynamics\), \"docs\", \"examples\", \"ieee39data\"\)"=>"DATA_DIR = raw\"$data\""); mod=Module(Symbol("SAMPLEWISE_",rand(UInt)))
 Core.eval(mod,:(using PowerDynamics,PowerDynamics.Library,ModelingToolkitBase,NetworkDynamics,OrdinaryDiffEqRosenbrock,OrdinaryDiffEqNonlinearSolve,CSV,DataFrames)); redirect_stdout(devnull) do; Base.include_string(mod,txt,"samplewise_part1.jl"); end
 nw=getfield(mod,:nw)
 Core.eval(mod, quote
  formula=@initformula :ZIPLoad₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
  set_initformula!($nw[VIndex(31)],formula)
  set_initformula!($nw[VIndex(39)],formula)
 end)
 pf=Base.invokelatest(solve_powerflow,nw;verbose=false); s0=Base.invokelatest(initialize_from_pf!,nw;pfs=pf,verbose=false); return nw,s0
end
function callbacks!(nw,buses)
 refs=Dict{Int,Ref{Float64}}();
 for b in buses
  r=Ref(0.0); refs[b]=r; init=Ref(false); bp=Ref(0.0); bq=Ref(0.0); syms=[Symbol("ZIPLoad₊Pset"),Symbol("ZIPLoad₊Qset")]
  aff=ComponentAffect([],syms) do u,p,ctx; if !init[]; bp[]=float(p[syms[1]]); bq[]=float(p[syms[2]]); init[]=true; end; p[syms[1]]=bp[]*(1+r[]); p[syms[2]]=bq[]*(1+r[]); end
  set_callback!(nw[VIndex(b)],PresetTimeComponentCallback([2.0],aff))
 end; refs
end
function runone(nw0,s00,r,dir)
 sp=split(String(r.support),"-"); buses=[parse(Int,x) for x in sp if x!="0"]; nw=deepcopy(nw0); s0=deepcopy(s00); refs=callbacks!(nw,buses); refs[buses[1]][]=Float64(r.amplitude_i); length(buses)>1 && (refs[buses[2]][]=Float64(r.amplitude_j)); path=joinpath(dir,String(r.point_id)*".csv"); isfile(path)&&return (true,0.0,path,"CHECKPOINT")
 t0=time(); ok=false; ret="ERROR"; try
  sol=solve(ODEProblem(nw,s0,(0.0,6.1)),Rodas5P();abstol=1e-11,reltol=1e-11,saveat=1/30,dtmax=1/60,maxiters=10^7); ret=string(sol.retcode); ok=ret=="Success"&&all(isfinite,sol.u[end]); rows=DataFrame(time=Float64[],bus=Int[],V_re=Float64[],V_im=Float64[])
  for t in sol.t,b in 1:39; z=complex(float(sol(t;idxs=VIndex(b,:busbar₊u_r))),float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); push!(rows,(t,b,real(z),imag(z))); end; ok&&CSV.write(path,rows)
 catch err; ret=sprint(showerror,err); ok=false; end; (ok,time()-t0,path,ret)
end
function main()
 pts=CSV.read(POINTS,DataFrame); pts=pts[pts.new_tds .== true,:]; mf=joinpath(RES,"tds_execution_manifest.csv"); out=isfile(mf) ? CSV.read(mf,DataFrame) : DataFrame(point_id=String[],op_tag=String[],status=String[],path=String[],runtime_s=Float64[],retcode=String[],sha256=String[])
 for (op,m) in (("op_m035",.35),("op_m085",.85),("op_m125",1.25))
  pp=pts[pts.op_tag .== op,:]; nrow(pp)==0&&continue; nw,s0=build(joinpath(ROOT,"output","t120_multi_op_independent_validation_v1","op_data",op)); dir=joinpath(PHYS,op,"results"); mkpath(dir)
  for r in eachrow(pp); x=runone(nw,s0,r,dir); h=isfile(x[3]) ? bytes2hex(sha256(read(x[3]))) : ""; !(String(r.point_id) in out.point_id)&&push!(out,(String(r.point_id),op,x[1] ? "EXECUTED_SUCCESS" : "EXECUTED_FAIL",x[3],x[2],x[4],h)); CSV.write(mf,out); println(op," ",r.point_id," ",x[1]," ",x[2]," ",x[4]); flush(stdout); end; GC.gc()
 end; CSV.write(mf,out)
end
main()
