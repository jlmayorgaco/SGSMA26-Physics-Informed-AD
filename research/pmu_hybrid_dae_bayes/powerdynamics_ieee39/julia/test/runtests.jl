using Test, CSV, DataFrames, LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA = joinpath(ROOT, "output", "provenance", "powerdynamics_original")

@testset "frozen IEEE39 inventory" begin
    bus = CSV.read(joinpath(DATA, "bus.csv"), DataFrame)
    branch = CSV.read(joinpath(DATA, "branch.csv"), DataFrame)
    load = CSV.read(joinpath(DATA, "load.csv"), DataFrame)
    machine = CSV.read(joinpath(DATA, "machine.csv"), DataFrame)
    @test nrow(bus) == 39
    @test nrow(branch) == 46
    @test nrow(load) == 19
    @test nrow(machine) == 10
    @test sort(bus.bus) == collect(1:39)
    @test all((branch.src_bus .>= 1) .& (branch.src_bus .<= 39) .& (branch.dst_bus .>= 1) .& (branch.dst_bus .<= 39))
end

@testset "independent Ybus structure" begin
    bus = CSV.read(joinpath(DATA, "bus.csv"), DataFrame)
    branch = CSV.read(joinpath(DATA, "branch.csv"), DataFrame)
    Y = zeros(ComplexF64, nrow(bus), nrow(bus))
    for r in eachrow(branch)
        i, j = Int(r.src_bus), Int(r.dst_bus)
        y = inv(ComplexF64(r.R, r.X))
        tap = iszero(r.r_src) ? 1.0 : Float64(r.r_src)
        Y[i,i] += (y + ComplexF64(r.G_src, r.B_src))*tap^2
        Y[j,j] += y + ComplexF64(r.G_dst, r.B_dst)
        Y[i,j] += -y*tap; Y[j,i] += -y*tap
    end
    @test size(Y) == (39, 39)
    @test count(x -> !iszero(x), Y) == 131
    @test maximum(abs.(Y - transpose(Y))) < 1e-12
end
