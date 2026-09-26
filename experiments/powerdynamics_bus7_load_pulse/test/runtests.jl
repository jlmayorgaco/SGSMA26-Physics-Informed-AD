using Test
using PowerDynamicsBus7LoadPulse

const B = PowerDynamicsBus7LoadPulse.Bus7LoadPulse

@testset "IEEE39 Bus-7 load pulse contract" begin
    @test length(B.EXPORT_TIMES) == 451
    @test B.EXPORT_TIMES[1] == 0.0
    @test B.EXPORT_TIMES[end] == 15.0
    @test all(B.EXPORT_TIMES .≈ collect(0.0:(1.0 / 30):15.0))

    expected_columns = [
        "TIMESTAMP",
        "BUS7_VA_ANG", "BUS7_VA_MAG", "BUS7_VB_ANG", "BUS7_VB_MAG",
        "BUS7_VC_ANG", "BUS7_VC_MAG", "BUS7_IA_ANG", "BUS7_IA_MAG",
        "BUS7_IB_ANG", "BUS7_IB_MAG", "BUS7_IC_ANG", "BUS7_IC_MAG",
        "BUS7_Freq", "BUS7_ROCOF", "DATA_PRESENT", "Event",
    ]
    @test B.raw_columns(7) == expected_columns
    @test B.raw_columns(1)[2] == "BUS1_VA_ANG"
    @test B.raw_columns(39)[end] == "Event"

    @test B.timestamp_projection([0.0, 1 / 30, 2 / 30, 3 / 30, 15.0]) == [0.0, 0.033, 0.066, 0.1, 15.0]
    @test B.event_labels([0.0, 4.999999, 5.0, 9.999999, 10.0, 15.0]) == [0, 0, 4, 4, 0, 0]

    command = B.event_command_frame(-2.0, -0.8, [4.999999, 5.0, 9.999999, 10.0])
    @test command.load_multiplier == [1.0, 1.1, 1.1, 1.0]
    @test command.P_command == [-2.0, -2.2, -2.2, -2.0]
    @test command.Q_command ≈ [-0.8, -0.88, -0.88, -0.8]
end

if get(ENV, "PD39_RUN_INTEGRATION_TEST", "0") == "1"
    mktempdir() do output_dir
        result = PowerDynamicsBus7LoadPulse.run_scenario(; output_dir, make_plots=false)
        @test result == output_dir
    end
end
