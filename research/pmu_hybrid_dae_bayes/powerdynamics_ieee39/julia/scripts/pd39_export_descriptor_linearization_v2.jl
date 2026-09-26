using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CSV, DataFrames
using LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output", "load_tangent_v2")
mkpath(joinpath(OUT, "results")); mkpath(joinpath(OUT, "manifests"))
ok = true; err = ""
try
    redirect_stdout(devnull) do
        include(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part2.jl"))
    end
    desc = linearize_network(s0)
    M = Matrix(desc.M)
    A = Matrix(desc.A)
    syms = SII.variable_symbols(nw)
    CSV.write(joinpath(OUT, "results", "descriptor_mass_matrix.csv"), DataFrame(M, :auto))
    CSV.write(joinpath(OUT, "results", "descriptor_A_matrix.csv"), DataFrame(A, :auto))
    rows = DataFrame(index=Int[], symbol=String[], mass=Float64[], kind=String[])
    for (i, s) in enumerate(syms)
        push!(rows, (i, string(s), Float64(M[i, i]), iszero(M[i, i]) ? "algebraic" : "differential"))
    end
    CSV.write(joinpath(OUT, "results", "descriptor_metadata_native.csv"), rows)
    open(joinpath(OUT, "manifests", "descriptor_export_status.txt"), "w") do io
        println(io, "ok=true")
        println(io, "variables=", length(syms))
        println(io, "differential=", count(!iszero, diag(M)))
        println(io, "algebraic=", count(iszero, diag(M)))
        println(io, "mass_shape=", size(M))
        println(io, "A_shape=", size(A))
    end
catch e
    ok = false; err = sprint(showerror, e, catch_backtrace())
end
if !ok
    open(joinpath(OUT, "manifests", "descriptor_export_status.txt"), "w") do io
        println(io, "ok=false"); println(io, err)
    end
end
println("descriptor_export_ok=", ok)
!ok && println(err)
