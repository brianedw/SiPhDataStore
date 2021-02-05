All of the following were placed in the same directory:
* kernel1.gds
* kernel2.gds
* kernel3.gds
* kernel4.gds
* Cal1.gds
* Cal2.gds
* Cal3.gds
* Cal4.gds
* LumericalGDSImportScript.lsf
* LumericalResultExportScript.lsf
* sim.fsp (empty simulation consisting of only an s-parameter sweep)

The file `sim.fsp` was opened in Lumerical FDTD (2019b).  Both lsf scripts were opened within Lumerical.

The following steps were then followed.
* Within `LumericalGDSImportScript.lsf` the `target` and `mode` values were set and the script was run.
* Within the S-Parameter Sweep, only the "input" ports were made to be active.  This halves the simulation time.
* The S-Parameter Sweep was "run".
* The `LumericalResultExportScript.lsf` script was run.
* The exported file (`SResults.txt`) was renamed.
* New values of `target` and `mode` were set and the process repeated.