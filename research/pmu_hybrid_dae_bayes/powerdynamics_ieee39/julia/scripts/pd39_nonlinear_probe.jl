using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CairoMakie
using LinearAlgebra
const ROOT = normpath(joinpath(@__DIR__, "..", "..")); const OUT = joinpath(ROOT, "output")
const EXAMPLE = joinpath(pkgdir(PowerDynamics), "docs", "examples")
ok=true; err=""
try
 redirect_stdout(devnull) do
   include(joinpath(EXAMPLE,"ieee39_part2.jl"))
 end
 desc=linearize_network(s0); red=reduce_dae(desc); vals,V=eigen(red.A)
 candidates=[i for i in eachindex(vals) if abs(vals[i]) > 1e-7 && real(vals[i]) >= -0.1]
 M=Matrix(desc.M); d=diag(M); didx=findall(!iszero,d); aidx=findall(iszero,d)
 x0=uflat(s0); p0=pflat(s0); eps=1e-4; probes=NamedTuple[]
 for idx in candidates
  vd=real(V[:,idx]); vd ./= norm(vd)
  vf=zeros(Float64,length(d)); vf[didx]=vd; vf[aidx]=-(desc.A[aidx,aidx]\(desc.A[aidx,didx]*vd))
  sp=NWState(nw,x0+eps*vf,p0,0.0)
  sol=solve(ODEProblem(nw,sp,(0.0,2.0)),Rodas5P(); abstol=1e-9,reltol=1e-9)
  push!(probes,(idx=idx,sol=sol))
 end
 open(joinpath(OUT,"results","pd_nonlinear_probe.txt"),"w") do io
   println(io,"probe_count=",length(probes))
   for q in probes
    sol=q.sol; idx=q.idx
    println(io,"mode_index=",idx," retcode=",sol.retcode," steps=",length(sol.t)," finite=",all(isfinite,sol.u[end])," mode_real=",real(vals[idx])," mode_imag=",imag(vals[idx]))
   end
 end
catch e
 ok=false; err=sprint(showerror,e,catch_backtrace()); open(joinpath(OUT,"results","pd_nonlinear_probe.txt"),"w") do io println(io,"retcode=ERROR\n",err) end
end
println("nonlinear_probe_ok=",ok); !ok && println(err)
