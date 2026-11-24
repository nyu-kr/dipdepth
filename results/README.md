# Results

The per-night values behind the paper's Tables II to IV and the numbers in its text. `python3 tables.py` rebuilds Tables II to IV and the text figures drawn from these rows, and checks them against `expected.json`. An empty cell is an undefined value. Rows are in the order the published analyses resampled them: the bootstrap draws row positions, so this order reproduces the intervals exactly.

## Conventions
- The expected drop is a rate per 10 s, also in a night's last bin when that bin is shorter.
- In the per-10-minute correlations a window is 60 whole bins of 10 s counted from the start of the recording; the last bin may be short, so the last window can lack up to 9 s; an event at the last window's end counts in it. The breathing-effort windows are whole 600 s windows, with belt validity counted on the smoothed envelopes.
- One Bengaluru recording is released twice (`same_recording_as`). Both copies stay in every new-hospital statistic, as registered; `tables.py` checks that dropping either copy changes no new-hospital verdict.

## sealed.csv: the 653 readable sealed nights
- `key`: dataset and record name
- `hospital`: Boston, Athens or Dublin
- `ahi`: scored apneas and hypopneas per hour of sleep, one decimal, the value the split was stratified on
- `scored_events`: scored respiratory events, as the dataset annotates them
- `converged`: both state fits converged, the expected drop's and the count-only one the model fits beside it (False on the two nights whose event counts the amendment kept)
- `count_events_f1`: events F1 of DipDepth's event count
- `rule_events_f1`: events F1 of the 3% rule's dips
- `count_per_10min_rho`: rank correlation across the night's 10-minute windows of DipDepth's events and the scored events
- `rule_per_10min_rho`: the same for the rule's dips
- `expected_drop_per_10min_rho`: the same for the expected drop averaged over each window
- `count_per_hour`: DipDepth's events per hour of recording
- `rule_per_hour`: the rule's dips per hour of recording
- `expected_drop_mean`: the night's mean expected drop per 10 s bin, SpO2 points
- `hypoxic_burden`: hypoxic burden, %min per hour of sleep
- `rule_summed_dips_per_hour`: the rule's dips, each by its fall, summed per hour of recording, SpO2 points
- `effort_expected_drop`: breathing effort, the rank correlation across 10-minute windows of the expected drop with belt variability
- `effort_pulse_depths`: the same for the summed depths of the runs of pulses (empty on the two unconverged nights)
- `effort_area_below_level`: the same for the summed drop below the starting usual level (empty on the two unconverged nights)
- `effort_summed_dips`: the same for the rule's dips, each by its fall (empty on the two unconverged nights)
- `effort_count`: the same for the rule's count

## unscored.csv: the 989 Boston nights without public scoring
- `key`: dataset and record name
- `excluded`: "prediction failed" (4 nights) or "belt not eligible" (1); empty on the 984 compared
- `effort_expected_drop`, `effort_pulse_depths`, `effort_area_below_level`, `effort_summed_dips`, `effort_count`: as in sealed.csv
- `ablation_order`: the night's row position in the ablation, which resampled the same nights in another order

## new_hospitals.csv: the 102 recordings from Jaén, Bengaluru and Udine
- `key`: dataset and record name
- `hospital`: Jaén, Bengaluru or Udine
- `excluded`: "no valid SpO2" (1), "model fit failed" (1) or "belt not eligible" (1)
- `same_recording_as`: the other copy of the one recording released twice
- `ahi`: scored apneas and hypopneas per hour of valid SpO2
- `scored_events`, `count_events_f1`, `rule_events_f1`, `count_per_10min_rho`, `rule_per_10min_rho`, `count_per_hour`, `rule_per_hour`: as in sealed.csv
- `effort_expected_drop`, `effort_pulse_depths`, `effort_area_below_level`, `effort_summed_dips`, `effort_count`: as in sealed.csv

## parameters.csv: the fitted model on the 653 readable sealed nights (Table IV)
- `key`: dataset and record name, in the order of sealed.csv
- `fall_s`, `recovery_s`: the night's fitted drop shape, fall time T_f and recovery time constant T_r in seconds; 20 and 20 on the 52 nights whose shape kept its start
- `pulses_per_hour`: pulses above 0.001 point per hour of recording
- `runs_per_hour`: runs of 1 point or more per hour of recording
- `state_step_sd`: the state's fitted step standard deviation sigma (empty on the two unconverged nights)

## split.csv: the 1,305 scored nights from Boston, Athens and Dublin
- `key`: dataset and record name
- `hospital`: Boston, Athens or Dublin
- `part`: build, check or test

## expected.json
- `summaries`: every value `tables.py` rebuilds, unrounded, as the registered analyses reported it; numbers must agree to 1e-9, counts and verdicts exactly
- `paper text, 3 places`: the duplicate-recording sensitivity, which only the paper's text reports: nights, the difference to 3 places and the verdict
