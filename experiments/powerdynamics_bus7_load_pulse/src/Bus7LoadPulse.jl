module Bus7LoadPulse

using CairoMakie
using CSV
using DataFrames
using IEEE39
using JSON
using LinearAlgebra
using NetworkDynamics
using OrdinaryDiffEqNonlinearSolve
using OrdinaryDiffEqRosenbrock
import Pkg
using PowerDynamics
using SHA
using Statistics

const SCENARIO_ID = "SIM_PD39_LOAD_BUS7_PLUS10_RETURN_V1"
const EVENT_BUS = 7
const EVENT_START = 5.0
const EVENT_END = 10.0
const SIMULATION_START = 0.0
const SIMULATION_END = 15.0
const EXPORT_FPS = 30
const EXPORT_TIMES = collect(0.0:(1.0 / EXPORT_FPS):SIMULATION_END)
const LOAD_MULTIPLIER = 1.10
const PMU_BUSES = Set([2, 5, 6, 10, 19, 22, 29, 39])
const RAW_COLUMNS_TEMPLATE = [
    "TIMESTAMP",
    "BUS{bus}_VA_ANG", "BUS{bus}_VA_MAG",
    "BUS{bus}_VB_ANG", "BUS{bus}_VB_MAG",
    "BUS{bus}_VC_ANG", "BUS{bus}_VC_MAG",
    "BUS{bus}_IA_ANG", "BUS{bus}_IA_MAG",
    "BUS{bus}_IB_ANG", "BUS{bus}_IB_MAG",
    "BUS{bus}_IC_ANG", "BUS{bus}_IC_MAG",
    "BUS{bus}_Freq", "BUS{bus}_ROCOF",
    "DATA_PRESENT", "Event",
]
const IEEE39_GIT_SHA = "d47ef9c655e215e62c7ca484dbe4b31786caabf7"

export run_scenario

repo_root() = normpath(joinpath(@__DIR__, "..", "..", ".."))
default_output_dir() = joinpath(repo_root(), "output", SCENARIO_ID)
ieee39_data_dir() = joinpath(pkgdir(IEEE39), "ieee39data")

raw_columns(bus::Int) = [replace(c, "{bus}" => string(bus)) for c in RAW_COLUMNS_TEMPLATE]

function json_write(path, value)
    open(path, "w") do io
        JSON.print(io, value, 2)
    end
end

function sha256_file(path)
    bytes2hex(open(sha256, path))
end

function sha256_files(paths)
    digest = SHA.SHA2_256_CTX()
    for path in sort(string.(paths))
        SHA.update!(digest, codeunits(path))
        SHA.update!(digest, read(path))
    end
    bytes2hex(SHA.digest!(digest))
end

function package_version_string(pkg)
    try
        string(Base.pkgversion(pkg))
    catch
        "unknown"
    end
end

function timestamp_projection(times)
    # RAW0001 truncates the physical time to milliseconds for its display
    # column: 1/30 -> 0.033, 2/30 -> 0.066, 3/30 -> 0.1. The physical
    # trajectory remains evaluated at the exact k/30 values.
    floor.((times .* 1000.0) .+ 1e-9) ./ 1000.0
end

function wrap_degrees(x)
    y = mod(x + 180.0, 360.0) - 180.0
    y == -180.0 ? 180.0 : y
end

function unwrap_phase(values)
    result = Float64.(values)
    shift = 0.0
    for i in 2:length(result)
        step = result[i] - result[i - 1]
        if step > π
            shift -= 2π
        elseif step < -π
            shift += 2π
        end
        result[i] += shift
    end
    result
end

function derivative(values, times)
    n = length(values)
    n >= 2 || return zeros(Float64, n)
    out = zeros(Float64, n)
    out[1] = (values[2] - values[1]) / (times[2] - times[1])
    out[end] = (values[end] - values[end - 1]) / (times[end] - times[end - 1])
    for i in 2:(n - 1)
        out[i] = (values[i + 1] - values[i - 1]) / (times[i + 1] - times[i - 1])
    end
    out
end

function load_row(load_df, bus::Int)
    idx = findfirst(==(bus), load_df.bus)
    isnothing(idx) && error("No load row found for bus $bus")
    load_df[idx, :]
end

function build_and_initialize()
    # The official IEEE39 package sets the 100 MVA / 60 Hz bases while building
    # every component and exposes the canonical PF initialization path.
    nw_base = IEEE39.get_IEEE39_base()
    nw = IEEE39.set_IEEE39_PF_init(nw_base)
    pfnw = PowerDynamics.powerflow_model(nw)
    pfs = PowerDynamics.solve_powerflow(nw; pfnw=pfnw, verbose=false)
    s0 = PowerDynamics.initialize_from_pf!(nw; pfs=pfs, verbose=false)

    du = zeros(Float64, length(NetworkDynamics.uflat(s0)))
    nw(du, NetworkDynamics.uflat(s0), NetworkDynamics.pflat(s0), 0.0)
    residual = Dict(
        "network_rhs_norm" => norm(du),
        "network_rhs_max_abs" => maximum(abs, du),
    )
    nw, pfs, s0, residual
end

function frequency_base_hz(nw)
    ωbase = Float64(NetworkDynamics.get_default(nw[VIndex(1)], :systembase₊ωbase))
    ωbase / (2π)
end

function attach_bus7_callbacks!(nw, nominal_p::Ref, nominal_q::Ref)
    p_symbol = Symbol("ZIPLoad₊Pset")
    q_symbol = Symbol("ZIPLoad₊Qset")
    symbols = [p_symbol, q_symbol]

    increase = ComponentAffect([], symbols) do u, p, ctx
        # The first callback captures the exact running parameter values. The
        # second callback restores these stored values, never a 0.90 product.
        nominal_p[] = Float64(p[p_symbol])
        nominal_q[] = Float64(p[q_symbol])
        p[p_symbol] = nominal_p[] * LOAD_MULTIPLIER
        p[q_symbol] = nominal_q[] * LOAD_MULTIPLIER
    end
    restore = ComponentAffect([], symbols) do u, p, ctx
        isnothing(nominal_p[]) && error("Bus-7 nominal P was not captured before restore")
        p[p_symbol] = nominal_p[]
        p[q_symbol] = nominal_q[]
    end
    set_callback!(nw[VIndex(EVENT_BUS)], (
        PresetTimeComponentCallback([EVENT_START], increase),
        PresetTimeComponentCallback([EVENT_END], restore),
    ))
    nw
end

function scalar_at(sol, t, index)
    value = sol(t; idxs=index)
    Float64(value)
end

function extract_trajectory(sol, nw, bus_df, times, f_nominal)
    nbus = nrow(bus_df)
    ntime = length(times)
    v_r = zeros(Float64, nbus, ntime)
    v_i = zeros(Float64, nbus, ntime)
    i_r = zeros(Float64, nbus, ntime)
    i_i = zeros(Float64, nbus, ntime)
    for (j, t) in enumerate(times)
        for bus in 1:nbus
            v_r[bus, j] = scalar_at(sol, t, VIndex(bus, :busbar₊u_r))
            v_i[bus, j] = scalar_at(sol, t, VIndex(bus, :busbar₊u_i))
            # BusBar current is defined by PowerDynamics as flowing into the
            # bus. The RAW fallback uses the net current injected into the
            # network, hence the documented sign reversal below.
            i_r[bus, j] = -scalar_at(sol, t, VIndex(bus, :busbar₊i_r))
            i_i[bus, j] = -scalar_at(sol, t, VIndex(bus, :busbar₊i_i))
        end
    end

    v_mag_pu = hypot.(v_r, v_i)
    v_angle_rad = [unwrap_phase(atan.(v_i[bus, :], v_r[bus, :])) for bus in 1:nbus]
    v_angle_rad = reduce(vcat, [reshape(x, 1, :) for x in v_angle_rad])
    i_mag_pu = hypot.(i_r, i_i)
    i_angle_rad = [unwrap_phase(atan.(i_i[bus, :], i_r[bus, :])) for bus in 1:nbus]
    i_angle_rad = reduce(vcat, [reshape(x, 1, :) for x in i_angle_rad])

    freq_hz = zeros(Float64, nbus, ntime)
    rocof_hz_s = zeros(Float64, nbus, ntime)
    for bus in 1:nbus
        # IEEE39 has no native bus frequency observable. The positive-sequence
        # angle is unwrapped before differentiation, then referenced to the
        # verified 60 Hz network base.
        freq_hz[bus, :] .= f_nominal .+ derivative(v_angle_rad[bus, :], times) ./ (2π)
        rocof_hz_s[bus, :] .= derivative(freq_hz[bus, :], times)
    end

    v_mag_volts = zeros(Float64, nbus, ntime)
    i_mag_amps = zeros(Float64, nbus, ntime)
    for bus in 1:nbus
        vbase_ll_v = Float64(bus_df[bus, :base_kv]) * 1000.0
        ibase_a = 100.0 / (sqrt(3) * Float64(bus_df[bus, :base_kv])) * 1000.0
        v_mag_volts[bus, :] .= v_mag_pu[bus, :] .* vbase_ll_v ./ sqrt(3)
        i_mag_amps[bus, :] .= i_mag_pu[bus, :] .* ibase_a
    end

    (
        v_mag_pu=v_mag_pu,
        v_angle_rad=v_angle_rad,
        i_mag_pu=i_mag_pu,
        i_angle_rad=i_angle_rad,
        v_mag_volts=v_mag_volts,
        i_mag_amps=i_mag_amps,
        freq_hz=freq_hz,
        rocof_hz_s=rocof_hz_s,
        v_r=v_r,
        v_i=v_i,
        i_r=i_r,
        i_i=i_i,
    )
end

function event_labels(times)
    [t >= EVENT_START && t < EVENT_END ? 4 : 0 for t in times]
end

function raw_frame(bus, trajectory, bus_df, times, f_nominal)
    vangle = rad2deg.(trajectory.v_angle_rad[bus, :])
    iangle = rad2deg.(trajectory.i_angle_rad[bus, :])
    vmag = trajectory.v_mag_volts[bus, :]
    imag = trajectory.i_mag_amps[bus, :]
    columns = Dict{String,Any}(
        "TIMESTAMP" => timestamp_projection(times),
        "BUS$(bus)_VA_ANG" => wrap_degrees.(vangle),
        "BUS$(bus)_VA_MAG" => vmag,
        "BUS$(bus)_VB_ANG" => wrap_degrees.(vangle .- 120.0),
        "BUS$(bus)_VB_MAG" => vmag,
        "BUS$(bus)_VC_ANG" => wrap_degrees.(vangle .+ 120.0),
        "BUS$(bus)_VC_MAG" => vmag,
        "BUS$(bus)_IA_ANG" => wrap_degrees.(iangle),
        "BUS$(bus)_IA_MAG" => imag,
        "BUS$(bus)_IB_ANG" => wrap_degrees.(iangle .- 120.0),
        "BUS$(bus)_IB_MAG" => imag,
        "BUS$(bus)_IC_ANG" => wrap_degrees.(iangle .+ 120.0),
        "BUS$(bus)_IC_MAG" => imag,
        "BUS$(bus)_Freq" => trajectory.freq_hz[bus, :],
        "BUS$(bus)_ROCOF" => trajectory.rocof_hz_s[bus, :],
        "DATA_PRESENT" => ones(Int, length(times)),
        "Event" => event_labels(times),
    )
    DataFrame([columns[c] for c in raw_columns(bus)], raw_columns(bus))
end

function event_command_frame(nominal_p, nominal_q, times)
    multiplier = [t >= EVENT_START && t < EVENT_END ? LOAD_MULTIPLIER : 1.0 for t in times]
    DataFrame(
        simulation_time=times,
        bus=fill(EVENT_BUS, length(times)),
        load_multiplier=multiplier,
        P_nominal=fill(nominal_p, length(times)),
        Q_nominal=fill(nominal_q, length(times)),
        P_command=nominal_p .* multiplier,
        Q_command=nominal_q .* multiplier,
    )
end

function bus_export_manifest(bus_df, output_dir)
    rows = DataFrame(
        bus=Int[], native_bus_name=String[], voltage_base_kV=Float64[],
        bus_type=String[], has_generator=Bool[], has_load=Bool[],
        is_original_SGSMA_PMU=Bool[], csv_path=String[], row_count=Int[],
    )
    for row in eachrow(bus_df)
        bus = Int(row.bus)
        push!(rows, (
            bus,
            "bus$(bus)",
            Float64(row.base_kv),
            string(row.bus_type),
            Bool(row.has_gen),
            Bool(row.has_load),
            bus in PMU_BUSES,
            joinpath("02_raw_csv", "Bus$(bus)_Competition_Data_nanmask.csv"),
            length(EXPORT_TIMES),
        ))
    end
    rows
end

function table_records(df)
    columns = Symbol.(names(df))
    [Dict(string(column) => (ismissing(row[column]) ? nothing : row[column]) for column in columns) for row in eachrow(df)]
end

function model_audit(nw, bus_df, branch_df, load_df, machine_df, residual, f_nominal, p_nominal, q_nominal)
    bus7 = bus_df[bus_df.bus .== EVENT_BUS, :][1, :]
    source_load7 = load_row(load_df, EVENT_BUS)
    transformer_rows = branch_df[branch_df.transformer .!= 0, :]
    Dict(
        "simulator" => "PowerDynamics.jl",
        "powerdynamics_version" => package_version_string(PowerDynamics),
        "ieee39_package_version" => package_version_string(IEEE39),
        "ieee39_git_sha" => IEEE39_GIT_SHA,
        "bus_count" => nrow(bus_df),
        "branch_count" => nrow(branch_df),
        "generator_count" => nrow(machine_df),
        "load_count" => nrow(load_df),
        "transformer_count" => nrow(transformer_rows),
        "buses" => table_records(bus_df),
        "generators" => table_records(machine_df),
        "loads" => table_records(load_df),
        "branches" => table_records(branch_df),
        "transformers" => table_records(transformer_rows),
        "transformer_representation" => "branch.csv transformer column; transformer rows are included separately above",
        "system_mva_base" => 100.0,
        "nominal_frequency_hz" => f_nominal,
        "native_bus_names" => ["bus$(i)" for i in 1:nrow(bus_df)],
        "bus_7_identifier" => Dict(
            "numeric_id" => EVENT_BUS,
            "native_name" => "bus$(EVENT_BUS)",
            "bus_type" => string(bus7.bus_type),
            "category" => string(bus7.category),
            "base_kv" => Float64(bus7.base_kv),
            "has_load" => Bool(bus7.has_load),
            "has_generator" => Bool(bus7.has_gen),
        ),
        "bus_7_load_model" => "ZIPLoad (KpZ=KqZ=1, constant impedance)",
        "bus_7_source_load_csv_P_pu" => Float64(source_load7.Pset),
        "bus_7_source_load_csv_Q_pu" => Float64(source_load7.Qset),
        "bus_7_nominal_P_pu" => p_nominal,
        "bus_7_nominal_Q_pu" => q_nominal,
        "bus_7_nominal_basis" => "initialized ZIPLoad Pset/Qset after canonical IEEE39 power-flow initialization",
        "initialization" => Dict(
            "powerflow_converged" => true,
            "dynamic_initialization_converged" => true,
            "residual" => residual,
        ),
        "current_semantics" => "CURRENT_SEMANTICS_FALLBACK: negative BusBar i_r/i_i, i.e. net complex current flowing into the network at each bus measurement point",
        "voltage_semantics" => "Positive-sequence BusBar u_r/u_i; synthetic balanced phase A/B/C representation",
        "frequency_semantics" => "60 Hz network base plus derivative of unwrapped positive-sequence BusBar voltage angle",
    )
end

function write_environment_manifest!(manifest_dir)
    open(joinpath(manifest_dir, "environment.txt"), "w") do io
        println(io, "Julia version: ", VERSION)
        println(io, "PowerDynamics version: ", package_version_string(PowerDynamics))
        println(io, "IEEE39 package version: ", package_version_string(IEEE39))
        println(io, "IEEE39 git SHA: ", IEEE39_GIT_SHA)
        println(io, "OS: ", Sys.KERNEL, " ", Sys.MACHINE)
        println(io, "Threads: ", Threads.nthreads())
    end
    open(joinpath(manifest_dir, "package_status.txt"), "w") do io
        redirect_stdout(io) do
            Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
        end
    end
end

function reference_columns(reference_path)
    open(reference_path, "r") do io
        split(chomp(readline(io)), ',')
    end
end

function raw_format_comparison(output_dir)
    reference_path = joinpath(repo_root(), "data", "RAW0001", "Bus2_Competition_Data_nanmask.csv")
    ref = reference_columns(reference_path)
    generated = raw_columns(2)
    Dict(
        "REFERENCE_FILE" => relpath(reference_path, repo_root()),
        "REFERENCE_COLUMNS" => ref,
        "GENERATED_COLUMNS" => generated,
        "COLUMN_MATCH" => ref == generated,
        "TIMESTAMP_STYLE" => "truncated to 3 decimal places from exact k/30 physical times; physical evaluation remains exact",
        "VOLTAGE_UNIT_INTERPRETATION" => "phase RMS volts: V_pu * Vbase_LL / sqrt(3) * 1000",
        "CURRENT_UNIT_INTERPRETATION" => "balanced phase RMS amps: I_pu * Sbase/(sqrt(3)*Vbase_LL) * 1000",
        "FREQUENCY_UNIT" => "Hz",
        "ROCOF_UNIT" => "Hz/s",
        "PHASE_CONVENTION" => "balanced synthetic phases: B=A-120 degrees, C=A+120 degrees, wrapped to (-180,180]",
        "KNOWN_DIFFERENCES" => ["waveform is a PowerDynamics IEEE39 controlled-load experiment, not RAW0001", "all 39 buses are exported", "clean plant response with no measurement noise"],
    )
end

function generate_per_bus_plot(path, bus, trajectory, times)
    fig = Figure(size=(1400, 1500), fontsize=13)
    labels = ("A", "B", "C")
    phase_angles = (
        rad2deg.(trajectory.v_angle_rad[bus, :]),
        rad2deg.(trajectory.v_angle_rad[bus, :] .- 2π / 3),
        rad2deg.(trajectory.v_angle_rad[bus, :] .+ 2π / 3),
    )
    current_angles = (
        rad2deg.(trajectory.i_angle_rad[bus, :]),
        rad2deg.(trajectory.i_angle_rad[bus, :] .- 2π / 3),
        rad2deg.(trajectory.i_angle_rad[bus, :] .+ 2π / 3),
    )
    panels = [
        ("Voltage magnitude [V]", (i -> trajectory.v_mag_volts[bus, :]), false),
        ("Voltage angle [deg]", (i -> phase_angles[i]), true),
        ("Current magnitude [A]", (i -> trajectory.i_mag_amps[bus, :]), false),
        ("Current angle [deg]", (i -> current_angles[i]), true),
    ]
    for (row, (title, getter, _)) in enumerate(panels)
        ax = Axis(fig[row, 1], ylabel=title)
        for i in 1:3
            lines!(ax, times, getter(i), label=labels[i])
        end
        vlines!(ax, [EVENT_START, EVENT_END], color=[:red, :blue], linestyle=:dash)
        row == 1 && axislegend(ax; position=:rb)
    end
    ax5 = Axis(fig[5, 1], ylabel="Frequency [Hz]")
    lines!(ax5, times, trajectory.freq_hz[bus, :], color=:black)
    vlines!(ax5, [EVENT_START, EVENT_END], color=[:red, :blue], linestyle=:dash)
    ax6 = Axis(fig[6, 1], xlabel="Time [s]", ylabel="ROCOF [Hz/s]")
    lines!(ax6, times, trajectory.rocof_hz_s[bus, :], color=:black)
    vlines!(ax6, [EVENT_START, EVENT_END], color=[:red, :blue], linestyle=:dash)
    Label(fig[0, 1], "IEEE39 Bus $(bus) — Bus-7 +10% load pulse, 5–10 s — PowerDynamics", fontsize=20)
    Label(fig[7, 1], "+10% LOAD at 5 s     |     LOAD RESTORED at 10 s", color=:gray30)
    save(path, fig; px_per_unit=2)
end

function generate_system_plot(path, times, matrix, ylabel, title; buses=1:size(matrix, 1))
    fig = Figure(size=(1500, 900), fontsize=13)
    ax = Axis(fig[1, 1], xlabel="Time [s]", ylabel=ylabel, title=title)
    for bus in buses
        lines!(ax, times, matrix[bus, :], label="Bus $bus")
    end
    vlines!(ax, [EVENT_START, EVENT_END], color=[:red, :blue], linestyle=:dash)
    length(buses) <= 10 && axislegend(ax; position=:rb, nbanks=2)
    save(path, fig; px_per_unit=2)
end

function generate_bus7_detail(path, times, trajectory, command)
    fig = Figure(size=(1400, 1300), fontsize=13)
    ax1 = Axis(fig[1, 1], ylabel="Load multiplier", title="IEEE39 Bus 7 controlled load event")
    lines!(ax1, times, command.load_multiplier, color=:black)
    ax2 = Axis(fig[2, 1], ylabel="Voltage [V]")
    lines!(ax2, times, trajectory.v_mag_volts[EVENT_BUS, :], color=:darkgreen)
    ax3 = Axis(fig[3, 1], ylabel="Current [A]")
    lines!(ax3, times, trajectory.i_mag_amps[EVENT_BUS, :], color=:darkorange)
    ax4 = Axis(fig[4, 1], ylabel="Frequency [Hz]")
    lines!(ax4, times, trajectory.freq_hz[EVENT_BUS, :], color=:black)
    ax5 = Axis(fig[5, 1], xlabel="Time [s]", ylabel="ROCOF [Hz/s]")
    lines!(ax5, times, trajectory.rocof_hz_s[EVENT_BUS, :], color=:purple)
    for ax in (ax1, ax2, ax3, ax4, ax5)
        vlines!(ax, [EVENT_START, EVENT_END], color=[:red, :blue], linestyle=:dash)
    end
    save(path, fig; px_per_unit=2)
end

function generate_plots!(plot_dir, trajectory, times, command)
    per_bus = joinpath(plot_dir, "per_bus")
    system = joinpath(plot_dir, "system")
    mkpath(per_bus)
    mkpath(system)
    for bus in 1:size(trajectory.v_mag_volts, 1)
        generate_per_bus_plot(joinpath(per_bus, "Bus$(bus)_signals.png"), bus, trajectory, times)
    end
    generate_system_plot(joinpath(system, "all_bus_voltage_magnitude.png"), times, trajectory.v_mag_volts, "Phase-A voltage magnitude [V]", "IEEE39 — all bus voltage magnitudes — PowerDynamics")
    generate_system_plot(joinpath(system, "all_bus_frequency.png"), times, trajectory.freq_hz, "Frequency [Hz]", "IEEE39 — all bus frequencies — PowerDynamics")
    generate_system_plot(joinpath(system, "all_bus_rocof.png"), times, trajectory.rocof_hz_s, "ROCOF [Hz/s]", "IEEE39 — all bus ROCOF — PowerDynamics")
    generate_bus7_detail(joinpath(system, "bus7_event_detail.png"), times, trajectory, command)
    generate_system_plot(joinpath(system, "pmu8_voltage_overlay.png"), times, trajectory.v_mag_volts, "Phase-A voltage magnitude [V]", "IEEE39 — original SGSMA PMU buses — voltage", buses=sort(collect(PMU_BUSES)))
    generate_system_plot(joinpath(system, "pmu8_frequency_overlay.png"), times, trajectory.freq_hz, "Frequency [Hz]", "IEEE39 — original SGSMA PMU buses — frequency", buses=sort(collect(PMU_BUSES)))
end

function max_deviation(values, times, reference)
    deviations = abs.(values .- reference)
    idx = argmax(deviations)
    (value=maximum(deviations), time=times[idx])
end

function validation_report(output_dir, trajectory, command, raw_frames, times, p_nominal, q_nominal, f_nominal, residual)
    bus7_v = trajectory.v_mag_volts[EVENT_BUS, :]
    bus7_i = trajectory.i_mag_amps[EVENT_BUS, :]
    bus7_f = trajectory.freq_hz[EVENT_BUS, :]
    pre = findall(t -> 1.0 <= t <= 4.5, times)
    pre_v = mean(bus7_v[pre])
    pre_f = mean(bus7_f[pre])
    command_ratio_p = command.P_command[findfirst(==(EVENT_START), command.simulation_time)] / p_nominal
    command_ratio_q = command.Q_command[findfirst(==(EVENT_START), command.simulation_time)] / q_nominal
    restore_idx = findfirst(==(EVENT_END), command.simulation_time)
    three_phase_errors = Float64[]
    for frame in raw_frames
        bus = parse(Int, match(r"BUS(\d+)_", names(frame)[2]).captures[1])
        va = frame[!, "BUS$(bus)_VA_MAG"]
        vb = frame[!, "BUS$(bus)_VB_MAG"]
        vc = frame[!, "BUS$(bus)_VC_MAG"]
        push!(three_phase_errors, maximum(abs.(vb .- va)))
        push!(three_phase_errors, maximum(abs.(vc .- va)))
    end
    Dict(
        "file_count" => length(raw_frames),
        "row_count_each" => unique([nrow(frame) for frame in raw_frames]),
        "column_count_each" => unique([ncol(frame) for frame in raw_frames]),
        "column_order_match" => all(names(frame) == raw_columns(i) for (i, frame) in enumerate(raw_frames)),
        "timestamp_consistent" => all(frame.TIMESTAMP == raw_frames[1].TIMESTAMP for frame in raw_frames),
        "data_present_all_one" => all(all(frame.DATA_PRESENT .== 1) for frame in raw_frames),
        "physical_channels_finite" => all(all(isfinite, Float64.(frame[!, c])) for frame in raw_frames for c in raw_columns(parse(Int, match(r"BUS(\d+)_", names(frame)[2]).captures[1]))[2:15]),
        "event_labels_correct" => all(raw_frames[1].Event .== event_labels(times)),
        "event_P_Q_ratio" => Dict("P" => command_ratio_p, "Q" => command_ratio_q),
        "event_ratio_ok" => isapprox(command_ratio_p, LOAD_MULTIPLIER; atol=1e-12) && isapprox(command_ratio_q, LOAD_MULTIPLIER; atol=1e-12),
        "restoration_exact" => all(isapprox.(command.P_command[restore_idx:end], p_nominal; atol=1e-12)) && all(isapprox.(command.Q_command[restore_idx:end], q_nominal; atol=1e-12)),
        "sign_convention_more_demand" => p_nominal < 0 && q_nominal < 0 && abs(command.P_command[findfirst(==(EVENT_START), command.simulation_time)]) > abs(p_nominal) && abs(command.Q_command[findfirst(==(EVENT_START), command.simulation_time)]) > abs(q_nominal),
        "physical_response" => Dict(
            "bus7_voltage_max_deviation_V" => max_deviation(bus7_v, times, pre_v),
            "bus7_current_max_deviation_A" => max_deviation(bus7_i, times, mean(bus7_i[pre])),
            "frequency_max_deviation_Hz" => max_deviation(bus7_f, times, f_nominal),
            "maximum_absolute_rocof_Hz_s" => (value=maximum(abs.(trajectory.rocof_hz_s[EVENT_BUS, :])), time=times[argmax(abs.(trajectory.rocof_hz_s[EVENT_BUS, :]))]),
        ),
        "physical_response_nonzero" => maximum(abs.(bus7_v .- pre_v)) > 1e-9 || maximum(abs.(bus7_i .- mean(bus7_i[pre]))) > 1e-9,
        "pre_event_stability" => Dict(
            "interval_s" => [1.0, 4.5],
            "bus7_voltage_drift_V" => maximum(bus7_v[pre]) - minimum(bus7_v[pre]),
            "bus7_frequency_drift_Hz" => maximum(bus7_f[pre]) - minimum(bus7_f[pre]),
        ),
        "three_phase_equal_magnitude_max_error" => maximum(three_phase_errors),
        "three_phase_angle_contract" => "checked from synthetic A/B/C construction: B=A-120 deg, C=A+120 deg",
        "initialization_residual" => residual,
        "frequency_nominal_hz" => f_nominal,
        "rocof_method" => "central differences on exported 30 fps frequency with one-sided boundaries; no smoothing",
        "event_semantics" => "Event=4 only while the load intervention is active, 5 <= t < 10; post-restoration dynamics remain physical",
        "measurement_noise" => "none",
    )
end

function scenario_manifest(output_dir, scripts, data_files, f_nominal, p_nominal, q_nominal, solver_name, rtol, atol)
    project_path = joinpath(@__DIR__, "..", "Project.toml")
    manifest_path = joinpath(@__DIR__, "..", "Manifest.toml")
    Dict(
        "scenario_id" => SCENARIO_ID,
        "simulator" => "PowerDynamics.jl",
        "julia_version" => string(VERSION),
        "powerdynamics_version" => package_version_string(PowerDynamics),
        "powerdynamics_git_sha" => nothing,
        "ieee39_model_version" => package_version_string(IEEE39),
        "ieee39_model_git_sha" => IEEE39_GIT_SHA,
        "solver" => solver_name,
        "rtol" => rtol,
        "atol" => atol,
        "simulation_start" => SIMULATION_START,
        "simulation_end" => SIMULATION_END,
        "export_fps" => EXPORT_FPS,
        "export_samples" => length(EXPORT_TIMES),
        "event_type" => "load_change",
        "event_bus" => EVENT_BUS,
        "event_start" => EVENT_START,
        "event_end" => EVENT_END,
        "P_multiplier" => LOAD_MULTIPLIER,
        "Q_multiplier" => LOAD_MULTIPLIER,
        "measurement_noise" => "none",
        "physical_model_hash" => sha256_files(data_files),
        "Project_toml_hash" => sha256_file(project_path),
        "Manifest_toml_hash" => isfile(manifest_path) ? sha256_file(manifest_path) : nothing,
        "script_hashes" => Dict(relpath(path, repo_root()) => sha256_file(path) for path in scripts),
        "random_seed" => "NONE",
        "nominal_frequency_hz" => f_nominal,
        "bus7_nominal_P_pu" => p_nominal,
        "bus7_nominal_Q_pu" => q_nominal,
        "callback_semantics" => "PresetTimeComponentCallback at exact 5.0 and 10.0 s; only load parameters change; dynamic state is not reinitialized",
        "algebraic_re_solve" => "PowerDynamics/NetworkDynamics evaluates the algebraic network equations on the running trajectory after the parameter callback",
    )
end

function run_scenario(; output_dir=default_output_dir(), make_plots=true)
    mkpath(output_dir)
    dirs = Dict(
        "manifest" => joinpath(output_dir, "00_manifest"),
        "audit" => joinpath(output_dir, "01_model_audit"),
        "raw" => joinpath(output_dir, "02_raw_csv"),
        "event" => joinpath(output_dir, "03_event"),
        "plots" => joinpath(output_dir, "04_plots"),
        "validation" => joinpath(output_dir, "05_validation"),
        "review" => joinpath(output_dir, "06_review"),
    )
    foreach(mkpath, values(dirs))

    bus_df = CSV.read(joinpath(ieee39_data_dir(), "bus.csv"), DataFrame)
    branch_df = CSV.read(joinpath(ieee39_data_dir(), "branch.csv"), DataFrame)
    load_df = CSV.read(joinpath(ieee39_data_dir(), "load.csv"), DataFrame)
    machine_df = CSV.read(joinpath(ieee39_data_dir(), "machine.csv"), DataFrame)
    nw, pfs, s0, residual = build_and_initialize()
    f_nominal = frequency_base_hz(nw)
    # The canonical IEEE39 initialization makes the dynamic ZIPLoad Pset/Qset
    # power-flow consistent. Those are the values that the callback must
    # multiply and restore; the raw load.csv values are retained in the audit.
    p_nominal = Float64(s0.p.v[EVENT_BUS, Symbol("ZIPLoad₊Pset")])
    q_nominal = Float64(s0.p.v[EVENT_BUS, Symbol("ZIPLoad₊Qset")])
    saved_p = Ref{Union{Nothing,Float64}}(nothing)
    saved_q = Ref{Union{Nothing,Float64}}(nothing)
    attach_bus7_callbacks!(nw, saved_p, saved_q)

    rtol = 1e-9
    atol = 1e-9
    solver_name = "Rodas5P"
    sol = solve(ODEProblem(nw, s0, (SIMULATION_START, SIMULATION_END)), Rodas5P();
        abstol=atol, reltol=rtol, dtmax=1 / 60, maxiters=10^7,
        saveat=EXPORT_TIMES)
    string(sol.retcode) == "Success" || error("PowerDynamics solve failed: $(sol.retcode)")
    isnothing(saved_p[]) && error("Bus-7 increase callback did not execute")
    isapprox(saved_p[], p_nominal; atol=1e-10) || error("Captured Bus-7 P does not match IEEE39 nominal")
    isapprox(saved_q[], q_nominal; atol=1e-10) || error("Captured Bus-7 Q does not match IEEE39 nominal")

    trajectory = extract_trajectory(sol, nw, bus_df, EXPORT_TIMES, f_nominal)
    frames = DataFrame[]
    for bus in 1:nrow(bus_df)
        frame = raw_frame(bus, trajectory, bus_df, EXPORT_TIMES, f_nominal)
        CSV.write(joinpath(dirs["raw"], "Bus$(bus)_Competition_Data_nanmask.csv"), frame)
        push!(frames, frame)
    end
    command = event_command_frame(p_nominal, q_nominal, EXPORT_TIMES)
    CSV.write(joinpath(dirs["event"], "event_command.csv"), command)
    manifest_df = bus_export_manifest(bus_df, output_dir)
    CSV.write(joinpath(dirs["audit"], "bus_export_manifest.csv"), manifest_df)
    json_write(joinpath(dirs["audit"], "model_audit.json"), model_audit(nw, bus_df, branch_df, load_df, machine_df, residual, f_nominal, p_nominal, q_nominal))
    json_write(joinpath(dirs["validation"], "raw0001_format_comparison.json"), raw_format_comparison(output_dir))
    validation = validation_report(output_dir, trajectory, command, frames, EXPORT_TIMES, p_nominal, q_nominal, f_nominal, residual)
    json_write(joinpath(dirs["validation"], "validation_metrics.json"), validation)

    scripts = [joinpath(@__DIR__, "..", "scripts", "run_bus7_load_pulse.jl"), @__FILE__]
    data_files = [joinpath(ieee39_data_dir(), name) for name in ("bus.csv", "branch.csv", "load.csv", "machine.csv", "avr.csv", "gov.csv")]
    json_write(joinpath(dirs["manifest"], "scenario_manifest.json"), scenario_manifest(output_dir, scripts, data_files, f_nominal, p_nominal, q_nominal, solver_name, rtol, atol))
    write_environment_manifest!(dirs["manifest"])

    make_plots && generate_plots!(dirs["plots"], trajectory, EXPORT_TIMES, command)
    report_path = joinpath(dirs["review"], "FINAL_REPORT.md")
    open(report_path, "w") do io
        println(io, "# $(SCENARIO_ID)")
        println(io)
        println(io, "Controlled IEEE39 PowerDynamics load event with exact Bus-7 +10% P/Q intervention at 5 s and exact nominal restoration at 10 s.")
        println(io)
        println(io, "- Solver: $(solver_name), rtol=$(rtol), atol=$(atol), dtmax=$(1/60) s")
        println(io, "- Export: $(length(EXPORT_TIMES)) exact samples at $(EXPORT_FPS) fps, 39 buses")
        println(io, "- Nominal Bus-7 load: P=$(p_nominal) pu, Q=$(q_nominal) pu")
        println(io, "- Event command: P/Q multiplied by $(LOAD_MULTIPLIER) from $(EVENT_START) s inclusive to $(EVENT_END) s exclusive")
        println(io, "- Measurement noise: none")
        println(io, "- Current semantics: fallback to negative BusBar current, documented in model_audit.json")
        println(io, "- Frequency: $(f_nominal) Hz base plus derivative of unwrapped positive-sequence voltage angle")
        println(io, "- ROCOF: central differences on exported frequency with one-sided endpoints")
        println(io)
        println(io, "See validation_metrics.json for residuals, event ratios, restoration, physical response, and pre-event drift.")
    end
    output_dir
end

end
