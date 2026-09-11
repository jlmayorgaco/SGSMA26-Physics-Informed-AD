using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra, CSV, DataFrames, Random, Statistics
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const EXAMPLE=joinpath(pkgdir(PowerDynamics),"docs","examples")
const OBS=[2,5,6,10,19,22,29,39]; const HIDDEN=setdiff(1:39,OBS); const TERMINALS=[(39,1),(29,26),(10,11),(22,21),(19,16),(2,1),(5,4),(6,5)]
ok=true; err=""
try
 redirect_stdout(devnull) do; include(joinpath(EXAMPLE,"ieee39_part2.jl")); end
 x0=uflat(s0); p0=pflat(s0); ps=SII.parameter_symbols(nw)
 gi=only(SII.parameter_index(nw,ps[findfirst(x->occursin("p_ref",lowercase(string(x))),ps)])); li=only(SII.parameter_index(nw,ps[findfirst(x->occursin("pset",lowercase(string(x))),ps)]))
 # Native edge indices for the eight requested terminals; orientation is explicit.
 edge_for=[11,37,18,35,31,1,8,5]; native_for=[false,false,true,true,true,false,false,false]
 function obsval(sol,t)
  z=Float64[]
  for b in OBS; push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_r)))); push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); end
  for (e,native) in zip(edge_for,native_for)
   if native; push!(z,float(sol(t;idxs=EIndex(e,:src₊i_r)))); push!(z,float(sol(t;idxs=EIndex(e,:src₊i_i))))
   else; push!(z,float(sol(t;idxs=EIndex(e,:dst₊i_r)))); push!(z,float(sol(t;idxs=EIndex(e,:dst₊i_i)))); end
  end; z
 end
 function hidval(sol,t)
  z=Float64[]; for b in HIDDEN; push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_r)))); push!(z,float(sol(t;idxs=VIndex(b,:busbar₊u_i)))); end; z
 end
 rng=MersenneTwister(20260911); ts=collect(range(0,3,length=91)); dataset=DataFrame()
 y0=obsval(ODEProblem(nw,NWState(nw,x0,p0,0.0),(0.0,0.01)) |> pr -> solve(pr,Rodas5P();abstol=1e-9,reltol=1e-9),0.0); h0=hidval(ODEProblem(nw,NWState(nw,x0,p0,0.0),(0.0,0.01)) |> pr -> solve(pr,Rodas5P();abstol=1e-9,reltol=1e-9),0.0)
 for split in ["DEV","TEST"]
  n=split=="DEV" ? 20 : 100
  for j in 1:n
   a=(0.002+0.008*rand(rng)); b=(0.002+0.008*rand(rng)); typ=mod(j,3)==0 ? "A+B" : mod(j,3)==1 ? "A_governor" : "B_load"; da=typ=="B_load" ? 0.0 : a; db=typ=="A_governor" ? 0.0 : b
   pp=copy(p0); pp[gi]*=(1+da); pp[li]*=(1+db); sol=solve(ODEProblem(nw,NWState(nw,x0,pp,0.0),(0.0,3.0)),Rodas5P();abstol=1e-9,reltol=1e-9)
   finite=all(isfinite,sol.u[end]); rc=string(sol.retcode)
   for (k,t) in enumerate(ts)
    ov=obsval(sol,t); hv=hidval(sol,t)
    vals=Any["$(split)_$(j)",split,typ,20260911+j,da,db,k-1,t,finite,rc]; append!(vals,ov); append!(vals,hv)
    colnames=Symbol.(vcat(["traj","split","scenario","seed","amplitude_a","amplitude_b","frame","time","finite","retcode"],["pmu_$(i)" for i in 1:length(ov)],["hidden_$(i)" for i in 1:length(hv)]))
    row=DataFrame(); for (nm,val) in zip(colnames,vals); row[!,nm]=[val]; end
    dataset=vcat(dataset,row;cols=:union)
   end
  end
 end
 CSV.write(joinpath(OUT,"results","e04_pd_dataset.csv"),dataset); CSV.write(joinpath(OUT,"results","e04_pd_y0.csv"),DataFrame(pmu=y0)); CSV.write(joinpath(OUT,"results","e04_pd_hidden0.csv"),DataFrame(hidden=h0))
 open(joinpath(OUT,"manifests","e04_pd_dataset_status.txt"),"w") do io; println(io,"ok=true"); println(io,"dev=20"); println(io,"test=100"); println(io,"frames=91"); println(io,"duration_s=3.0"); end
catch e; global ok=false; global err=sprint(showerror,e,catch_backtrace()); end
open(joinpath(OUT,"manifests","e04_pd_dataset_status.txt"),"w") do io; println(io,"ok=",ok); println(io,err); end
println("e04_pd_dataset_ok=",ok); !ok && println(err)
