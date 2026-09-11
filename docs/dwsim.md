# DWSIM connection experiment

The bundled **Humid Air** flowsheet has been exercised through DWSIM 10.2.6
on macOS ARM64. Python prepares a finite batch; DWSIM's built-in IronPython
interpreter changes the feed, solves the flowsheet and exports JSON; ordinary
Python checks the returned observations.

This is a repository experiment, not yet a public `DWSIMAdapter`. It requires
one manual execution in DWSIM's Script Manager. Operange's package dependencies
are unchanged, and no DWSIM installation, assembly or sample is redistributed.

## Run the example

From an Operange checkout, create a new run directory:

```bash
python -m examples.dwsim_humid_air prepare /tmp/operange-dwsim-run
```

The default sample is
`/Applications/DWSIM.app/Contents/MacOS/samples/Humid Air.dwxml`.
Use `--sample /path/to/Humid\ Air.dwxml` for another installation. Only the
bundled Humid Air model is supported by this experiment; other flowsheets and
platforms require separate validation.

1. Open the **copy** identified by the command in DWSIM.
2. Open **Tools → Script Manager**, select **IronPython**, and paste the four
   lines printed by `prepare` into a new script. Leave **Run on event** off.
3. Click **Run Async** once and wait for completion. Do not edit the flowsheet
   or launch another calculation during the batch.
4. Check the exported observations:

```bash
python -m examples.dwsim_humid_air check /tmp/operange-dwsim-run
python -m examples.dwsim_humid_air check /tmp/operange-dwsim-run --capacity-kw 4
```

The run directory contains the copied flowsheet, copied runner, launcher,
`request.json` and `response.json`. Preparation refuses to overwrite an existing
directory. Use a new directory for a new study. Keep the full directory to
preserve the request, response and executable used for that study.

The runner restores the original feed and recalculates after the batch. It does
not save changes back to the flowsheet file. The checker exits nonzero for a
missing response, protocol error, unresolved calculation or failed restoration.
An observed capacity violation is a completed calculation and does not by itself
make the command fail.

## What was observed

The sample cools humid air to its dew point using Peng–Robinson thermodynamics.
It fixes outlet vapor fraction at 1, cooler efficiency at 100%, and pressure
drop at zero. The inlet is 101325 Pa, with its bundled composition unchanged.

A live run on 11 September 2026 returned:

| Case | Inlet temperature | Mass flow | Cooling duty | Margin against an illustrative 2.5 kW limit |
| --- | --- | --- | --- | --- |
| Nominal | 25 °C | 0.280556 kg/s | 2.371945 kW | +0.128055 kW |
| Warmer feed | 30 °C | 0.280556 kg/s | 3.805951 kW | −1.305951 kW |
| Higher flow | 25 °C | 0.3366672 kg/s | 2.846333 kW | −0.346333 kW |
| Nominal repeated | 25 °C | 0.280556 kg/s | 2.371945 kW | +0.128055 kW |

All four cases returned an outlet temperature of approximately **16.727386 °C**.
All objects were calculated without reported errors. Recomputed mass and energy
balance residuals were zero at exported precision, and the repeated nominal
case reproduced the original duty. The feed was restored successfully.

The 2.5 kW and 4 kW capacities are **illustrative downstream limits**, not
equipment ratings read from the sample. Raising this reporting limit to 4 kW
puts all four observed duties within it; it does not simulate a redesigned cooler.

## Evidence and limits

The reader checks the run ID, request digest, copied flowsheet and runner
digests, scenario coverage, actual feed values, calculation flags, object errors,
outlet vapor fraction, pressure drop, and finite numerical values. It independently
recomputes total mass and enthalpy balances from exported streams and duty, with
absolute tolerances of `1e-8 kg/s` and `1e-7 kW` respectively. DWSIM's enthalpy
and energy API values use **kJ/kg and kW**; temperature and pressure use K and Pa.

Results record the runtime version, property-package type and a hash of the
serialized in-memory state before the batch. This distinguishes the state from
the source-file bytes, but the hash alone cannot reconstruct unsaved edits.
For this experiment, start from the prepared copy without manual model changes.

These checks establish consistent observations from this simulator and operating
specification. They do not independently validate the thermodynamic model,
establish that alternative controls cannot recover service, or certify an
unlisted point in a continuous uncertainty domain. Nonconvergence and missing
properties remain **unresolved**. No `process_result/v1` robustness certificate
is emitted by this experiment.

The offline tests inject solver errors, stale inputs, incomplete coverage,
changed files, missing values and broken balances. They use synthetic data and
do not substitute for the manual DWSIM smoke test.

## Recommended first supported adapter

Start with an isolated simulator worker that loads a fresh copy for each
realization, applies explicitly mapped inputs and fixed operating settings,
solves with a timeout, and returns named outputs plus diagnostics. Bind the
flowsheet, compounds, property-package configuration, solver settings and
runtime version to model provenance. Add direct evaluation and finite scenario
audits first, preserving unresolved cases and limiting violations to the
declared fixed operating rule. Continuous robustness, adjustable recourse,
failure-distance guarantees and portable simulator controllers need further
evidence machinery.

DWSIM 10 has an upstream [standalone MCP server](https://github.com/DanWBR/dwsim10/tree/v10.2.6/tools/DWSIM.MCPServer)
that is a candidate for this worker boundary. The installed macOS 10.2.6 desktop
bundle does not contain that server or `DWSIM.Automation.dll`; the
[10.2.6 release assets](https://github.com/DanWBR/dwsim10/releases/tag/v10.2.6)
provide MCP packages for Linux. A supported macOS headless build and its failure
handling still need to be proven before selecting that transport. The desktop
AI assistant's MCP connections are a separate client feature.

The working experiment instead uses DWSIM's
[built-in script manager](https://github.com/DanWBR/dwsim10/blob/v10.2.6/ui/DWSIM.UI.Desktop.Avalonia/ScriptEditorWindow.axaml.cs)
and synchronous
[`RequestCalculationAndWait`](https://github.com/DanWBR/dwsim10/blob/v10.2.6/engine/DWSIM.FlowsheetBase/FlowsheetBase.vb)
API. This proves the installed calculation engine can exchange useful data with
Python before committing the library to a transport or extra runtime dependency.
