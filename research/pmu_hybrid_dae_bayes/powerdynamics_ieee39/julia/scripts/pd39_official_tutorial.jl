using PowerDynamics
using PowerDynamics.Library
using ModelingToolkitBase
using NetworkDynamics
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using CairoMakie

const OUT = normpath(joinpath(@__DIR__, "..", "..", "output"))
const EXAMPLE = joinpath(pkgdir(PowerDynamics), "docs", "examples")
const DATA = joinpath(EXAMPLE, "ieee39data")

function run_part(label, file)
    started = time()
    ok = true
    err = ""
    try
        mod = Module(Symbol("PowerDynamicsIEEE39_", label))
        Core.eval(mod, :(using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CairoMakie))
        Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
        Base.include(mod, joinpath(EXAMPLE, file))
    catch e
        ok = false
        err = sprint(showerror, e, catch_backtrace())
    end
    open(joinpath(OUT, "manifests", "$(label)_stdout.txt"), "w") do io
        println(io, "label=", label, "\nok=", ok, "\nelapsed_s=", time()-started, "\nerror=", err)
    end
    return ok, err
end

println("PowerDynamics pkgdir: ", pkgdir(PowerDynamics))
println("data directory: ", DATA)
for (label, file) in (("part1", "ieee39_part1.jl"), ("part2", "ieee39_part2.jl"), ("part3", "ieee39_part3.jl"))
    ok, err = run_part(label, file)
    println(label, " ok=", ok)
    !ok && println(err)
end
