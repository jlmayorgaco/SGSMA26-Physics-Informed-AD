# Advanced PMU Signal Analysis Report

## 1. Scope

This report was generated automatically from the raw SGSMA/IEEE-39 PMU CSV files. It extends the original plotting-only workflow with advanced exploratory signal analysis, event-centric transient inspection, frequency-domain analysis, Hilbert-envelope diagnostics, filtering views, missing-data quantification, and cross-bus correlation/overlay analysis.

## 2. Inputs

- Raw directory: `C:\Users\walla\Documents\Github\SGSMA26-Physics-Informed-AD\V3\data\raw`
- PMU metadata file: `None`
- Timeline file: `None`
- Buses analyzed: Bus10, Bus19, Bus2, Bus22, Bus29, Bus39, Bus5, Bus6

## 3. Important methodological note

The available signals are PMU phasor features sampled at roughly 30 fps, not high-frequency instantaneous voltage/current waveforms. Therefore, the FFT, spectral peaks, Hilbert transform, and 'harmonic-like' outputs in this report characterize low-frequency PMU dynamics, envelopes, oscillatory modes, transient signatures, and inter-area/electromechanical behaviour visible in the phasor-domain time series. They do **not** represent classical waveform harmonic estimation of the 60 Hz instantaneous signal.

## 4. Output structure

- `Bus*/plots/`: overviews, histograms, spectra, noise diagnostics, Hilbert plots, filter-bank plots
- `Bus*/event_zoom/`: high-resolution event-centered transient zooms
- `Bus*/stats/`: JSON summaries for all channels
- `combined/`: cross-bus overlays, correlations, missing-data matrix, combined summary

## 5. Combined multi-PMU findings

- Highest missing-data ratio: **Bus29** with ratio `0.057009`.
- Cross-bus overlay rankings for `Frequency`, `ROCOF`, and `VA_mag` were generated to help identify which PMU exhibits the strongest normalized excursion during each event.
- See `combined/corr_frequency.png`, `combined/corr_rocof.png`, and `combined/missing_matrix.png` for the principal joint diagnostics.

## 6. Per-bus summary

### Bus10

- Descriptor: `Bus10`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `114582.77598527016` to `223160.7442423454`; peak-to-peak: `108577.96825707523`
- `IA_mag` range: `279.6836530905095` to `1307.9479390006031`; peak-to-peak: `1028.2642859100936`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.059185606042023636, 0.06658380679727659, 0.07398200755252954, 0.014796401510505909]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.7028290717490306, 0.7102272725042836, 0.6954308709937778, 0.7176254732595366, 0.7250236740147895]`
- Estimated `Frequency` SNR [dB]: `6.090950679119505`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-12.77166532398935`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus19

- Descriptor: `Bus19`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `147789.56511203438` to `236054.786556697`; peak-to-peak: `88265.22144466263`
- `IA_mag` range: `380.4409834143511` to `1502.8305856007216`; peak-to-peak: `1122.3896021863707`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.059185606042023636, 0.06658380679727659, 0.07398200755252954, 0.014796401510505909]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.6954308709937778, 0.6880326702385248, 0.7028290717490306, 0.6806344694832718, 0.7102272725042836]`
- Estimated `Frequency` SNR [dB]: `5.9986831936023055`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-15.986415074795545`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus2

- Descriptor: `Bus2`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `76832.99359` to `229173.4910339168`; peak-to-peak: `152340.4974439168`
- `IA_mag` range: `37.99095865986117` to `2163.5659851753903`; peak-to-peak: `2125.5750265155293`
- Dominant `Frequency` spectral peaks [Hz]: `[0.007398200756272444, 0.07398200756272444, 0.066583806806452, 0.05918560605017955, 0.014796401512544887]`
- Dominant `ROCOF` spectral peaks [Hz]: `[1.3094815338602226, 1.3168797346164949, 1.30208333310395, 1.2946851323476776, 1.3242779353727674]`
- Estimated `Frequency` SNR [dB]: `4.062483697099739`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-14.699151972454194`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `7` (Bad data): `3.266s` to `3.966s`, duration `0.700s`
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus22

- Descriptor: `Bus22`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `148692.17297465733` to `233530.067409182`; peak-to-peak: `84837.89443452467`
- `IA_mag` range: `574.0824352514461` to `1741.8432405620777`; peak-to-peak: `1167.7608053106314`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.059185606042023636, 0.06658380679727659, 0.07398200755252954, 0.014796401510505909]`
- Dominant `ROCOF` spectral peaks [Hz]: `[9.662050186360359, 9.654651985605106, 9.647253784849852, 9.669448387115612, 9.6398555840946]`
- Estimated `Frequency` SNR [dB]: `6.0156936688431575`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-21.100184607042213`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus29

- Descriptor: `Bus29`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.057009`; gap count: `4`; max gap: `191.400000 s`
- `VA_mag` range: `-1684.476101262685` to `245224.8434869307`; peak-to-peak: `246909.3195881934`
- `IA_mag` range: `-7.380544672751394` to `832.9729065065849`; peak-to-peak: `840.3534511793363`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.07398200755252954, 0.06658380679727659, 0.059185606042023636, 0.0813802083077825]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.0073982007552529545, 0.014796401510505909, 0.022194602265758864, 1.006155302714402, 0.9987571019591489]`
- Estimated `Frequency` SNR [dB]: `5.904656595392068`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-2.010840500891273`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `5` (Missing data): `536.566s` to `563.600s`, duration `27.034s`
  - Event `5` (Missing data): `614.333s` to `643.400s`, duration `29.067s`
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `5` (Missing data): `2584.666s` to `2776.066s`, duration `191.400s`
  - Event `5` (Missing data): `2935.766s` to `2976.333s`, duration `40.567s`
  - Event `6` (Missing data + physical event): `2976.366s` to `2994.800s`, duration `18.434s`
  - Event `3` (Generation change/outage): `2994.833s` to `3293.933s`, duration `299.100s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus39

- Descriptor: `Bus39`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `49787.88395920708` to `229770.2406113245`; peak-to-peak: `179982.35665211742`
- `IA_mag` range: `46.10951094126492` to `2491.4689518973173`; peak-to-peak: `2445.3594409560524`
- Dominant `Frequency` spectral peaks [Hz]: `[0.007398200756272444, 0.066583806806452, 0.05918560605017955, 0.07398200756272444, 0.08138020831899688]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.24414062495699065, 0.2367424242007182, 0.2515388257132631, 0.22934422344444574, 0.25893702646953554]`
- Estimated `Frequency` SNR [dB]: `6.531099291809631`; normality p-value of residual: `3.746151358007908e-42`
- Estimated `ROCOF` SNR [dB]: `2.887976250989036`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `7` (Bad data): `334.366s` to `335.100s`, duration `0.734s`
  - Event `7` (Bad data): `336.533s` to `337.566s`, duration `1.033s`
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus5

- Descriptor: `Bus5`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `110833.0971855405` to `219680.5489918537`; peak-to-peak: `108847.4518063132`
- `IA_mag` range: `539.0709459005556` to `1467.8291969809534`; peak-to-peak: `928.7582510803978`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.06658380679727659, 0.059185606042023636, 0.07398200755252954, 0.014796401510505909]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.7028290717490306, 0.7102272725042836, 0.7176254732595366, 0.6954308709937778, 0.6880326702385248]`
- Estimated `Frequency` SNR [dB]: `6.07552567367004`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-10.722542202907846`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

### Bus6

- Descriptor: `Bus6`
- Sampling rate: `30.3030` fps
- Rows: `161379`
- Missing ratio: `0.000000`; gap count: `0`; max gap: `0.000000 s`
- `VA_mag` range: `112327.78131131864` to `220270.514179924`; peak-to-peak: `107942.73286860537`
- `IA_mag` range: `362.7994240237559` to `1571.112913328118`; peak-to-peak: `1208.3134893043623`
- Dominant `Frequency` spectral peaks [Hz]: `[0.0073982007552529545, 0.06658380679727659, 0.059185606042023636, 0.07398200755252954, 0.014796401510505909]`
- Dominant `ROCOF` spectral peaks [Hz]: `[0.7028290717490306, 0.7102272725042836, 0.6954308709937778, 0.7176254732595366, 0.6880326702385248]`
- Estimated `Frequency` SNR [dB]: `6.097635936563464`; normality p-value of residual: `0.0`
- Estimated `ROCOF` SNR [dB]: `-10.519095261035607`; normality p-value of residual: `0.0`
- Detected abnormal spans:
  - Event `1` (Fault): `1173.166s` to `1178.100s`, duration `4.934s`
  - Event `2` (Line outage): `2375.233s` to `2379.400s`, duration `4.167s`
  - Event `3` (Generation change/outage): `2976.366s` to `3293.933s`, duration `317.567s`
  - Event `4` (Load change/drop): `3877.733s` to `4182.100s`, duration `304.367s`

## 7. Suggested next research extensions

1. Build event-onset detectors from the derivative/Hilbert/spectral features already saved in the JSON files.
2. Add topology-aware analysis using the RAW file to relate event magnitude with electrical distance.
3. Extend the combined overlays to estimate propagation delay or bus ranking confidence intervals.
4. Use the generated event windows as a deterministic feature-extraction stage for classifiers or sequence models.
