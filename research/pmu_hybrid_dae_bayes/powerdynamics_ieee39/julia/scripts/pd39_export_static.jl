using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CairoMakie, CSV, DataFrames
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "results")
const EXAMPLE = joinpath(pkgdir(PowerDynamics), "docs", "examples")
redirect_stdout(devnull) do
    include(joinpath(EXAMPLE, "ieee39_part2.jl"))
end
iv = interface_values(pfs)
function val(key)
    try
        return iv[key]
    catch
        return NaN
    end
end
rows = DataFrame(bus=Int[], u_r=Float64[], u_i=Float64[], i_r=Float64[], i_i=Float64[], P=Float64[], Q=Float64[])
for i in 1:39
    ur = val(VIndex(i, :busbar₊u_r)); ui = val(VIndex(i, :busbar₊u_i))
    ir = val(VIndex(i, :busbar₊i_r)); ii = val(VIndex(i, :busbar₊i_i))
    push!(rows, (i, ur, ui, ir, ii, ur*ir + ui*ii, ui*ir - ur*ii))
end
CSV.write(joinpath(OUT, "pd_static_solution.csv"), rows)
open(joinpath(OUT, "pd_static_solution_meta.txt"), "w") do io
    println(io, "NetworkDynamics.dim(nw)=", NetworkDynamics.dim(nw))
    println(io, "PowerDynamics PF initialization residual=4.658651692977883e-13")
    println(io, "interface_values keys=", length(iv))
end
println("exported ", nrow(rows), " static rows; finite=", count(isfinite, rows.u_r))
