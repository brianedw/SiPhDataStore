# Experimental Data

Experimental data file names have the form:

* `K4_T31_n15dBm_44d5K.csv`
* `K4_T31_T41_n15d5dBm_44d5K.csv`
* `K4_T31_R_n15d5dBm_44d5K.csv`

They can be interpreted as

`K[kernel number]_[Measurement]_[IncidentPowerAtCoupler]_[TIAResistor].csv`

* Within `kernel number`, it can be any of the designed kernels ie [1, 2, 3, 4].
* Within `IncidentPowerAtCoupler`, `n` &rightarrow;`-` and `d`&rightarrow;`.`, such that `n15d5dBm`&rightarrow;`-15.5dBm`.
* Within `TIAResistor`, `d`&rightarrow;`.` and `K`&rightarrow;`kOhm`, such that `44d5K`&rightarrow;`44.5KOhm`
* Within `Measurement`,  we can have one of several options.
  * `T31` would be the power measurement when port 1 is illuminated  and a PD is on port 3.
  * `T31_T41` would be the power measurement when port 1 is illuminated  and a PD is on the interferred output from port 3 and port 4.
  * `T31_R` would be the power measurement when port 1 and the Reference WG are illuminated  and a PD is on the interferred output from port 3 and the Reference WG.