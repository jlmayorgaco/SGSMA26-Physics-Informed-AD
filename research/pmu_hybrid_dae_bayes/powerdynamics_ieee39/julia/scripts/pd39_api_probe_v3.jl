using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
function main()
    include(joinpath(pkgdir(PowerDynamics), "docs", "examples", "ieee39_part2.jl"))
    nw = NetworkDynamics.extract_nw(s0)
    ps = SII.parameter_symbols(s0)
    os = SII.observed_symbols(nw)
    println("nparams=", length(ps), " nobs=", length(os), " nvars=", length(SII.variable_symbols(nw)))
    for (i,p) in enumerate(ps)
        occursin("ZIPLoad", string(p)) && println("P ", i, " ", p)
    end
    println("OBS_FIRST")
    for (i,o) in enumerate(os[1:min(30,end)])
        println(i, " ", o)
    end
end
main()
