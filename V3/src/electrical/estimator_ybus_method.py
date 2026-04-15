"""
YBUS organizada = [[Y_OO  Y_OU]; [Y_UO  Y_UU]]

[I_O]   [Y_OO  Y_OU] [V_O]
[I_U] = [Y_UO  Y_UU] [V_U]

You have the full Ybus from the .raw file. At any timestep where all 8 PMUs report valid data, you have 8 voltage phasors and 8 current injections. The network equation I = Ybus · V has 39 unknowns (bus voltages), and you're measuring 8 of them directly. This is a classic linear state estimation problem, and with PMU data it's actually linear (not the nonlinear WLS of traditional SCADA state estimation — that's a huge advantage).
Concretely:
Partition buses into observed (O, the 8 PMU buses) and unobserved (U, the other 31). Write Ybus in block form:
[I_O]   [Y_OO  Y_OU] [V_O]
[I_U] = [Y_UO  Y_UU] [V_U]
At non-PMU buses, the current injection I_U is either zero (transit buses with no generation/load) or equals the known load/generation from the .raw file (up to slow drift). So:
V_U = Y_UU⁻¹ · (I_U − Y_UO · V_O)
This lets you reconstruct voltages at all 31 non-PMU buses at every timestep, using only the 8 measured voltages and known injections. This is massive for localization — you now have a "virtual PMU" at every bus.
How this wins you the competition:

Residual-based fault localization: Compute I_U from your reconstructed V using the full Ybus. If the residual |I_U,measured_injection − I_U,reconstructed| spikes at a specific non-PMU bus, that's where the fault is. You're essentially doing bad-data detection, but using it as event localization. This pins faults to exact buses even when they occur at non-PMU locations — which is the hardest part of Task 3.
Line outage detection via topology hypothesis testing: For each of the 34 lines, build a modified Ybus with that line removed. For each timestep after a detected event, check which modified Ybus minimizes the reconstruction residual. The best-fit topology tells you which line tripped. This is dramatically more accurate than pattern matching.
Filling missing-data frames (Label 5/6): When Bus 29's PMU drops, you can still estimate V_29 from the other 7 PMUs via the same equation. This means your classifier doesn't lose information during cyber events — you convert missing data into reconstructed data plus a "reconstructed" flag.

Watch out for: During faults, the network topology momentarily changes (fault impedance appears) and Ybus is no longer accurate. So residuals will spike — that's the feature, not a bug. The spike tells you a fault happened; its spatial pattern tells you where.

"""


def Instantaneous_Y_BUS_Estimator_(V_O, I_O):
    V_U = Y_UU⁻¹ · (I_U − Y_UO · V_O)
    return [ V_U, I_U]