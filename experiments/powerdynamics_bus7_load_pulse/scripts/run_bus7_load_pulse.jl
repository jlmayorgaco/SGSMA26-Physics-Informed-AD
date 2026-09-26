using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using PowerDynamicsBus7LoadPulse
using IEEE39
using PowerDynamics

output_dir = get(ENV, "PD39_BUS7_OUTPUT", PowerDynamicsBus7LoadPulse.Bus7LoadPulse.default_output_dir())
make_plots = lowercase(get(ENV, "PD39_BUS7_PLOTS", "true")) in ("1", "true", "yes")

println("Running $(PowerDynamicsBus7LoadPulse.Bus7LoadPulse.SCENARIO_ID)")
println("Output: ", output_dir)
println("Julia: ", VERSION)
println("PowerDynamics: ", PowerDynamicsBus7LoadPulse.Bus7LoadPulse.package_version_string(PowerDynamics))
println("IEEE39 SHA: ", PowerDynamicsBus7LoadPulse.Bus7LoadPulse.IEEE39_GIT_SHA)
PowerDynamicsBus7LoadPulse.run_scenario(; output_dir, make_plots)
println("Completed successfully")
