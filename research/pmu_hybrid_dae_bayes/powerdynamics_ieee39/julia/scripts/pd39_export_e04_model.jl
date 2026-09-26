using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using LinearAlgebra, CSV, DataFrames
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output"); const EXAMPLE=joinpath(pkgdir(PowerDynamics),"docs","examples")
ok=true; err=""
try
 redirect_stdout(devnull) do; include(joinpath(EXAMPLE,"ieee39_part2.jl")); end
 desc=linearize_network(s0); red=reduce_dae(desc); A=Matrix(red.A); CSV.write(joinpath(OUT,"results","e04_A.csv"),DataFrame(A,:auto))
 obs=[2,5,6,10,19,22,29,39]; hidden=setdiff(1:39,obs); pout=Any[]; hout=Any[]
 for b in obs; push!(pout,VIndex(b,:busbar₊u_r)); push!(pout,VIndex(b,:busbar₊u_i)); end
 # Keep the terminal-current convention identical to the nonlinear dataset:
 # native=true means the requested bus is the package src end; otherwise use dst.
 edge_for=[11,37,18,35,31,1,8,5]; native_for=[false,false,true,true,true,false,false,false]
 for (e,native) in zip(edge_for,native_for)
   side=native ? :src : :dst
   push!(pout,EIndex(e,Symbol(side,"₊i_r"))); push!(pout,EIndex(e,Symbol(side,"₊i_i")))
 end
 for b in hidden; push!(hout,VIndex(b,:busbar₊u_r)); push!(hout,VIndex(b,:busbar₊u_i)); end
 inp=VIndex(1,:busbar₊u_r); lp=reduce_dae(linearize_network(s0;in=inp,out=pout)); lh=reduce_dae(linearize_network(s0;in=inp,out=hout))
 CSV.write(joinpath(OUT,"results","e04_C_pmu.csv"),DataFrame(Matrix(lp.C),:auto)); CSV.write(joinpath(OUT,"results","e04_C_hidden.csv"),DataFrame(Matrix(lh.C),:auto))
 # Export the actual static operating-point output, not C*0 (which is a delta).
 iv=interface_values(pfs)
 y0=Float64[]
 for b in obs; push!(y0,float(iv[VIndex(b,:busbar₊u_r)])); push!(y0,float(iv[VIndex(b,:busbar₊u_i)])); end
 for (e,native) in zip(edge_for,native_for)
   side=native ? :src : :dst
   push!(y0,float(iv[EIndex(e,Symbol(side,"₊i_r"))])); push!(y0,float(iv[EIndex(e,Symbol(side,"₊i_i"))]))
 end
 CSV.write(joinpath(OUT,"results","e04_y0_pmu.csv"),DataFrame(value=y0))
 open(joinpath(OUT,"results","e04_model_meta.txt"),"w") do io; println(io,"state_dim=",size(A,1)); println(io,"pmu_channels=",size(lp.C,1)); println(io,"hidden_channels=",size(lh.C,1)); println(io,"sampling_hz=30"); end
catch e; global ok=false; global err=sprint(showerror,e,catch_backtrace()); end
open(joinpath(OUT,"manifests","e04_model_export_status.txt"),"w") do io; println(io,"ok=",ok); println(io,err); end
println("e04_model_export_ok=",ok); !ok && println(err)
