# Export the old analytic (instantaneous-consistency) state initialization at
# the first canonical post-event sample.  This is an audit reference only.
using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using CSV, DataFrames, Random, LinearAlgebra, DelimitedFiles
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
const ROOT=normpath(joinpath(@__DIR__,"..","..")); const OUT=joinpath(ROOT,"output","second_order_op_robustness_v1"); const SRC0=joinpath(@__DIR__,"pd39_analytic_second_order_v1.jl"); const PKG=joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39data"); const DT=1/30
const OPS=[("op_m035",.35),("op_m085",.85),("op_m125",1.25)]; const BUSES=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]; const PAIRS=[(BUSES[i],BUSES[j],i,j) for i in 1:length(BUSES)-1 for j in i+1:length(BUSES)]
function run(tag,m)
  data=mktempdir(OUT); for f in readdir(PKG); CSV.write(joinpath(data,f),CSV.read(joinpath(PKG,f),DataFrame)); end
  bus=CSV.read(joinpath(data,"bus.csv"),DataFrame); k=only(findall(Int.(bus.bus).==3)); bus[k,:P]=Float64(bus[k,:P])*(1+0.03*m); bus[k,:Q]=Float64(bus[k,:Q])*(1+0.03*m); CSV.write(joinpath(data,"bus.csv"),bus)
  outpath=joinpath(OUT,"analytic_"*tag*"_state"); srcpath=joinpath(data,"ieee39_part1.jl")
  txt=read(SRC0,String); txt=replace(txt,"const OUT = joinpath(ROOT, \"output\", \"analytic_second_order_dae_v1\")"=>"const OUT = raw\""*outpath*"\""); txt=replace(txt,"const SRC = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39_part1.jl\")"=>"const SRC = raw\""*srcpath*"\""); txt=replace(txt,"const NF = 30"=>"const NF = 30")
  part=read(joinpath(pkgdir(PowerDynamics),"docs","examples","ieee39_part1.jl"),String); part=replace(part,"DATA_DIR = joinpath(pkgdir(PowerDynamics), \"docs\", \"examples\", \"ieee39data\")"=>"DATA_DIR = raw\""*data*"\""); write(srcpath,part)
  # Emit full state derivatives after the k=2 (t=DT) analytic step.  The
  # exporter stores self terms as Q=1/2*u_ii, so state second derivatives are
  # multiplied by two for the comparison contract.
  needle="        for (r,(_,_,ii,jj)) in enumerate(pairlist)\n            x2a[:,r] .= -(A_aa \\ (A_ad*x2[:,r] + He[aidx,ii,jj]))"
  inj="""        if k == 2
            for b in 1:length(BUSES)
                vv = xf[:,b]
                CSV.write(joinpath(RES, \"state_first_self_\"*string(BUSES[b])*\".csv\"), DataFrame(reshape(vv,1,:), :auto))
            end
        end
""" * needle
  txt=replace(txt,needle=>inj)
  # Save each completed self state immediately after the algebraic solve.
  needle2="            qself[(k-1)*nout+1:k*nout,b] .= 0.5 .* (Cfull*full_state(x2self[:,b],x2aself[:,b]) + Hobse[:,b,b])"
  inj2="        if k == 2\n            CSV.write(joinpath(RES, \"state_second_self_\"*string(BUSES[b])*\".csv\"), DataFrame(reshape(full_state(x2self[:,b],x2aself[:,b]),1,:), :auto))\n        end\n" * needle2
  txt=replace(txt,needle2=>inj2)
  needle3="            qcross[(k-1)*nout+1:k*nout,r] .= Cfull*full_state(x2[:,r],x2a[:,r]) + Hobse[:,ii,jj]"
  # Export every cross direction at the first post-event sample.  The full
  # state-map closure requires all 120 current analytic initializations, not
  # just the pilot subset.
  inj3=needle3 * "\n            if k == 2\n                CSV.write(joinpath(RES, \"state_second_cross_\"*string(BUSES[ii])*\"_\"*string(BUSES[jj])*\".csv\"), DataFrame(reshape(full_state(x2[:,r],x2a[:,r]),1,:), :auto))\n            end"
  txt=replace(txt,needle3=>inj3)
  mod=Module(Symbol("ANINIT_",rand(UInt))); Base.include_string(mod,txt,"analytic_initial_state_$(tag).jl")
  println("analytic_initial_state_done ",tag)
end
for (tag,m) in OPS; run(tag,m); end
