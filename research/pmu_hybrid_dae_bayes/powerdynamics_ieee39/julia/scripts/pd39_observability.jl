using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using LinearAlgebra, CSV, DataFrames
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const EXAMPLE=joinpath(pkgdir(PowerDynamics),"docs","examples")
ok=true; err=""
try
 redirect_stdout(devnull) do
  include(joinpath(EXAMPLE,"ieee39_part2.jl"))
 end
 desc=linearize_network(s0); red=reduce_dae(desc); A=Matrix(red.A); n=size(A,1); Ad=exp(A/30.0)
 obs=[2,5,6,10,19,22,29,39]; outs=Any[]
 for b in obs; push!(outs,VIndex(b,:busbar₊u_r)); push!(outs,VIndex(b,:busbar₊u_i)); end
 # The package C matrix is built by the supported linearization API.
 input=VIndex(1,:busbar₊u_r); lsys=linearize_network(s0; in=input, out=outs); lred=reduce_dae(lsys); C=Matrix(lred.C)
 println("Csize=",size(C)," finite=",all(isfinite,C))
 if size(C,2)==0 || !all(isfinite,C); error("package output linearization returned a non-finite or zero-column C matrix; VI-only observability not reached (size=$(size(C)))"); end
 rows=DataFrame(horizon=Int[],rank=Int[],sigma_min=Float64[],sigma_max=Float64[],condition_number=Float64[],residual_norm=Float64[],functional_output_residual=Float64[],noise_information_bound_sigma1e3=Float64[],worst_hidden_bus=Int[])
 hidden=setdiff(collect(1:39),obs)
 targetouts=Any[]; for b in hidden; push!(targetouts,VIndex(b,:busbar₊u_r)); push!(targetouts,VIndex(b,:busbar₊u_i)); end
 target_sys=linearize_network(s0; in=input, out=targetouts); target_red=reduce_dae(target_sys); Ct=Matrix(target_red.C)
 for h in [0,3,10,30,60,120,180]
   blocks=[C*(Ad^k) for k in 0:h]; O=vcat(blocks...)
   # Use the observability Gramian spectrum to avoid a Windows LAPACK SVD
   # workspace failure for the tall h=180 matrix.
   gram=LinearAlgebra.Symmetric(O' * O); ee=LinearAlgebra.eigen(gram); order=sortperm(ee.values; rev=true); ev=real(ee.values[order]); U=ee.vectors[:,order]
   smax=isempty(ev) ? 0.0 : sqrt(max(ev[1],0.0)); tol=max(ev[1],0.0)*1e-9
   rk=count(>(tol),ev); pos=ev[ev .> tol]; smin=isempty(pos) ? 0.0 : sqrt(max(pos[end],0.0))
   res=rk==length(ev) ? 0.0 : sqrt(sum(max.(ev[rk+1:end],0.0)))
   P=rk==0 ? zeros(Float64,n,n) : U[:,1:rk]*U[:,1:rk]'
   fres=norm(Ct*(I-P)); per=[norm(Ct[2*k-1:2*k,:]*(I-P)) for k in eachindex(hidden)]; wi=isempty(pos) ? Inf : 1e-6*sum(1 ./ pos)
   push!(rows,(h,rk,smin,smax,smin>0 ? smax/smin : Inf,res,fres,wi,hidden[argmax(per)]))
 end
 CSV.write(joinpath(OUT,"results","pd_observability_horizons.csv"),rows)
 open(joinpath(OUT,"reports","pd_observability.md"),"w") do io
  println(io,"# PD functional observability (VI-only structural audit)\n")
  println(io,"- Observed buses: `",join(obs,", "), "`; outputs are rectangular voltage components only (no PMU data file is claimed).")
  println(io,"- Horizons tested: `0, 3, 10, 30, 60, 120, 180`; state dimension: `",n,"`; output rows: `",size(C,1),"`.")
  println(io,"- The table reports rank/singular spectrum of `[C; CA_d; ...; CA_d^h]` with `A_d=exp(A/30)` and functional residual of the 62-dimensional complex-voltage target map for the other 31 buses.")
  println(io,"- `noise_information_bound_sigma1e3` is the trace-style Gramian pseudoinverse bound assuming independent rectangular-voltage noise σ=1e-3 pu; it is a structural bound, not estimator performance.")
 end
catch e; global ok=false; global err=sprint(showerror,e,catch_backtrace()); open(joinpath(OUT,"reports","pd_observability.md"),"w") do io; println(io,"# PD functional observability (VI-only structural audit)\n\nSTATUS: NOT REACHED\n\n",err); end; end
open(joinpath(OUT,"manifests","observability_status.txt"),"w") do io; println(io,"ok=",ok); println(io,err); end
println("observability_ok=",ok); !ok && println(err)
