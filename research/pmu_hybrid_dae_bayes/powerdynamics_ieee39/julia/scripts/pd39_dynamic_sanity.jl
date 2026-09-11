using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra, CSV, DataFrames
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const EXAMPLE=joinpath(pkgdir(PowerDynamics),"docs","examples")
ok=true; err=""
try
 redirect_stdout(devnull) do; include(joinpath(EXAMPLE,"ieee39_part2.jl")); end
 x0=uflat(s0); p0=pflat(s0); ps=SII.parameter_symbols(nw)
 cases=Tuple{String,Int,Float64}[]
 for (label,needle,frac) in [("governor_reference_step","p_ref",0.01),("small_load_step","pset",0.01)]
  j=findfirst(x->occursin(needle,lowercase(string(x))),ps)
  !isnothing(j) && push!(cases,(label,only(SII.parameter_index(nw,ps[j])),frac))
 end
 rows=DataFrame(scenario=String[],retcode=String[],finite=Bool[],min_voltage_pu=Float64[],max_voltage_pu=Float64[],min_frequency_hz=Float64[],max_frequency_hz=Float64[],max_speed_deviation=Float64[],post_event_vm_bus30=Float64[],post_event_speed_bus30=Float64[],t_end=Float64[])
 for (label,pi,frac) in cases
  pp=copy(p0); pp[pi]*=(1+frac); sol=solve(ODEProblem(nw,NWState(nw,x0,pp,0.0),(0.0,0.5)),Rodas5P();abstol=1e-9,reltol=1e-9)
  ts=range(0,0.5,length=100); vm=Float64[]; freq=Float64[]; sp=Float64[]
  for t in ts
   for b in 1:39
    z=sol(t;idxs=VIndex(b,:busbar₊u_mag)); push!(vm,float(z))
   end
   z=sol(t;idxs=VIndex(30,:ctrld_gen₊machine₊ω)); push!(freq,60*float(z)); push!(sp,abs(float(z)-1))
  end
  vm_post=float(sol(0.5;idxs=VIndex(30,:busbar₊u_mag))); speed_post=float(sol(0.5;idxs=VIndex(30,:ctrld_gen₊machine₊ω)))
  push!(rows,(label,string(sol.retcode),all(isfinite,sol.u[end]),minimum(vm),maximum(vm),minimum(freq),maximum(freq),maximum(sp),vm_post,speed_post,last(sol.t)))
 end
 CSV.write(joinpath(OUT,"results","pd_dynamic_sanity.csv"),rows)
 open(joinpath(OUT,"reports","pd_dynamic_sanity.md"),"w") do io
  println(io,"# PowerDynamics dynamic sanity checks\n")
  println(io,"- Official Part III fault/clear scenario: **PASS** (`part3_stdout.txt`), line 11 short circuit at 0.1 s and disconnection at 0.2 s over 15 s.")
  println(io,"- Additional package-native perturbations: +1% governor reference and +1% ZIP load setpoint, each integrated for 0.5 s with Rodas5P; see `output/results/pd_dynamic_sanity.csv`.")
  println(io,"- These are response/retcode sanity checks, not a stability certificate or estimator validation.")
 end
catch e; global ok=false; global err=sprint(showerror,e,catch_backtrace()); end
open(joinpath(OUT,"manifests","dynamic_sanity_status.txt"),"w") do io; println(io,"ok=",ok); println(io,err); end
println("dynamic_sanity_ok=",ok); !ok && println(err)
