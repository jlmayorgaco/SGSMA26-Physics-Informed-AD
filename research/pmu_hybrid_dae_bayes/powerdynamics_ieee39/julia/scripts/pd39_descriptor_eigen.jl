using PowerDynamics, PowerDynamics.Library, ModelingToolkitBase, NetworkDynamics
using NetworkDynamics: SII
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, CairoMakie, CSV, DataFrames
using LinearAlgebra
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = joinpath(ROOT, "output")
const EXAMPLE = joinpath(pkgdir(PowerDynamics), "docs", "examples")
mkpath(joinpath(OUT, "results")); mkpath(joinpath(OUT, "reports"))
ok = true; err = ""
try
    redirect_stdout(devnull) do
        include(joinpath(EXAMPLE, "ieee39_part2.jl"))
    end
    desc = linearize_network(s0)
    red = reduce_dae(desc)
    md = Matrix(desc.M)
    d = diag(md)
    syms = SII.variable_symbols(nw)
    rows = DataFrame(index=Int[], symbol=String[], mass=Float64[], kind=String[], component=String[], bus=Union{Missing,Int}[], state_name=String[])
    for i in eachindex(syms)
        ss = string(syms[i])
        m = match(r"VIndex[(]([0-9]+),[ ]*:([^₊)]+)", ss)
        busid = isnothing(m) ? missing : parse(Int, m.captures[1])
        parts = split(ss, '₊')
        comp = isnothing(m) ? (isempty(parts) ? ss : first(parts)) : m.captures[2]
        st = isempty(parts) ? ss : replace(last(parts), ")" => "")
        push!(rows, (i, ss, Float64(d[i]), iszero(d[i]) ? "algebraic_zero_mass" : "differential", comp, busid, st))
    end
    CSV.write(joinpath(OUT, "results", "pd_descriptor_inventory.csv"), rows)
    vals = eigvals(red.A)
    ord = sortperm(vals; by=x -> (-real(x), abs(imag(x))))
    erows = DataFrame(mode=Int[], real=Float64[], imag=Float64[], frequency_hz=Float64[], damping_ratio=Float64[], magnitude=Float64[], classification=String[])
    for (k, idx) in enumerate(ord)
        λ = vals[idx]; mag = abs(λ); ζ = mag == 0 ? NaN : -real(λ)/mag
        cls = abs(λ) < 1e-7 ? "reference_or_zero_candidate" : "physical_candidate"
        push!(erows, (k, real(λ), imag(λ), imag(λ)/(2π), ζ, mag, cls))
    end
    CSV.write(joinpath(OUT, "results", "pd_eigenvalues.csv"), erows)
    physical = erows[erows.classification .== "physical_candidate", :]
    near = physical[physical.real .> -2, :]
    CSV.write(joinpath(OUT, "results", "pd_modes.csv"), near)
    pf = participation_factors(desc)
    prows = DataFrame(mode=Int[], eigenvalue_real=Float64[], eigenvalue_imag=Float64[], state_symbol=String[], participation=Float64[], state_family=String[])
    pord = sortperm(pf.eigenvalues; by=x -> (-real(x), abs(imag(x))))
    for (mode, idx) in enumerate(pord)
        order = sortperm(view(pf.pfactors, :, idx); rev=true)
        for k in order[1:min(8, length(order))]
            ss = string(pf.sym[k])
            fam = occursin("machine", lowercase(ss)) ? "machine" : occursin("avr", lowercase(ss)) ? "avr" : occursin("gov", lowercase(ss)) ? "governor" : occursin("busbar", lowercase(ss)) ? "network_bus" : "other"
            push!(prows, (mode, real(pf.eigenvalues[idx]), imag(pf.eigenvalues[idx]), ss, pf.pfactors[k, idx], fam))
        end
    end
    CSV.write(joinpath(OUT, "results", "pd_participation.csv"), prows)
    spectral = maximum(real.(vals))
    refcount = count(abs.(vals) .< 1e-7)
    open(joinpath(OUT, "reports", "pd_small_signal.md"), "w") do io
        println(io, "# PowerDynamics IEEE39 descriptor and small-signal audit\n")
        println(io, "- Descriptor shape: `", size(md), "`; rank(M): `", rank(md), "`; differential rows: `", count(!iszero, d), "`; zero-mass rows: `", count(iszero, d), "`." )
        println(io, "- Reduced ODE dimension: `", size(red.A, 1), "`; reference/zero candidates (|λ|<1e-7): `", refcount, "`." )
        println(io, "- Spectral abscissa: `", spectral, " s^-1`." )
        println(io, "- Stability classification: `", spectral < -1e-8 ? "STABLE" : spectral <= 1e-8 ? "MARGINALLY_STABLE_REFERENCE_ONLY" : "UNSTABLE", "` (based on reduced package-supported linearization)." )
        println(io, "- Reference-angle treatment: the package retains the full network state and does not manually delete a bus angle; algebraic elimination is performed by `reduce_dae` using the descriptor partition." )
        println(io, "- Full eigenvalue table: `output/results/pd_eigenvalues.csv`; modes with Re(λ)>-2 s^-1: `output/results/pd_modes.csv`; dominant participation rows: `output/results/pd_participation.csv`." )
    end
catch e
    ok = false; err = sprint(showerror, e, catch_backtrace())
end
open(joinpath(OUT, "manifests", "descriptor_eigen_status.txt"), "w") do io
    println(io, "ok=", ok)
    println(io, err)
end
println("descriptor_eigen_ok=", ok)
!ok && println(err)
