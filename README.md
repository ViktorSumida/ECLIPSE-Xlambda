# EXCALIBUR-Xλ: Transmission Spectra Simulation Accounting for Stellar Activity

**Author**: Viktor Y. D. Sumida  
**Contact**: viktor.sumida@outlook.com

---

## 🌟 Purpose

EXCALIBUR-Xλ is a simulation tool designed to model **exoplanetary transmission spectra** in the presence of **stellar activity**, such as **starspots and faculae**. It produces wavelength-dependent light curves and builds a **grid of transit depths (Dλ)**, which can be used in **retrievals** or spectral corrections.

This framework allows you to assess how the presence of active regions on a star can distort observed transmission spectra, especially for planets orbiting M dwarfs or active stars.

---

## 🧠 Concept

The user provides a set of stellar, planetary, and spot parameters. The simulation:

1. Creates synthetic stellar disks (with and without starspots/faculae).
2. Simulates the planetary transit using the `Eclipse` class.
3. Calculates light curves and extracts wavelength-dependent transit depths.
4. Outputs a transmission spectrum (`D_lambda`) at each wavelength (or filter).

---

## 📂 File Structure

- `main.py`: Main engine that runs the simulations and handles output.
- `interpolation.ipynb`: Jupyter notebook for setting key simulation parameters and initiating batch runs.
- `star.py`, `eclipse_nv1.py`, `Planeta.py`: Core modules to handle stellar disks, transits, and orbital mechanics.
- `verify.py`: Utilities for validation and auxiliary calculations.

---

## ⚙️ How It Works

Each simulation takes as input:

- **Stellar parameters**: Radius, temperature, limb darkening, spot temperature
- **Planetary parameters**: Radius, semi-major axis, inclination, eccentricity, anomaly
- **Spot configuration**: Number, size (`r`), latitude/longitude
- **Spectral grid**: `lambdaEff`, `c1-c4` (limb-darkening), max intensity, number of wavelengths

---

## 🧾 Key Parameters in `interpolation.ipynb`

| Parameter | Description |
|----------|-------------|
| `raioStar`, `massStar`, `tempStar` | Stellar radius [R☉], mass [M☉], and effective temperature [K] |
| `raioPlanetaRj` | Planetary radius in Jupiter radii [R_J] |
| `periodo`, `anguloInclinacao`, `semiEixoUA`, `ecc`, `anom` | Orbital parameters: period [days], inclination [deg], semi-major axis [AU], eccentricity, anomaly |
| `starspots`, `quantidade`, `lat`, `longt`, `r` | Starspot configuration: enable spots, number of spots, their latitude/longitude, and radius (fraction of R★) |
| `c1`, `c2`, `c3`, `c4` | Limb darkening coefficients (from 4-parameter law, e.g., ExoCTK) |
| `plot_anim`, `plot_graph`, `plot_star` | Flags for animation, light curve plotting, and stellar image visualization |
| `min_pixels`, `max_pixels`, `pixels_per_rp` | Controls for matrix resolution and size limits |
| `ff_spot_min`, `ff_spot_max`, `T_spot_min`, `T_spot_max` | Spot simulation ranges: filling factor and temperature contrast |
| `num_ff_spot_simulations`, `num_T_spot_simulations` | Number of starspot simulations across `ff` and `T` ranges |
| `ff_fac_min`, `ff_fac_max`, `T_fac_min`, `T_fac_max` | Facula simulation ranges |
| `num_ff_fac_simulations`, `num_T_fac_simulations` | Number of facula simulations across `ff` and `T` ranges |
| `num_ff_interpolations`, `num_T_spot_interpolations` | Grid resolution used for retrieval-ready interpolation of the simulation output |

---

## 🔧 Getting Limb Darkening Coefficients from ExoCTK

To generate the coefficients used in your simulations:

1. Go to [ExoCTK Limb Darkening Tool](https://exoctk.stsci.edu/limb_darkening)
2. Choose the **4-parameter law**
3. Set the stellar parameters (Teff, log(g), metallicity, etc.)
4. Export the result as `.txt`
5. Ensure the file includes the coefficients `c1`, `c2`, `c3`, `c4` per wavelength
6. The simulation will read the values automatically
7. Important note: If you'd like to use limb darkening coefficients derived from observational data, simply follow the same table format and input your own values for `c1`, `c2`, `c3`, `c4`, and `wave_eff`

---

## 📈 Output

- Light curves are plotted per wavelength.
- A `.txt` file is generated with the following columns:
  - `f_spot`: Filling factor
  - `tempSpot`: Spot temperature
  - `wavelength`: Wavelength (nm)
  - `D_lambda`: Transit depth at that wavelength

If `f_spot = 0`, the output is stored as a **spotless baseline**.

---

## 🧪 Use Case: Grid Generation for Retrievals

You can automate the generation of multiple transmission spectra (e.g., with different spot coverages or temperatures) using the `interpolation.ipynb` notebook as a launcher.

These grids can later be ingested by your retrieval codes to infer atmospheric or stellar parameters more accurately.

---

## 🛠 Requirements

- Python 3.8+
- `NumPy`, `Pandas`, `Matplotlib`, `Numba`
- Custom modules: `star.py`, `eclipse_nv1.py`, etc.

---

## 📊 Example Output

![image](https://github.com/user-attachments/assets/a02dddab-7b19-464e-a610-5747da7dce18)

---

## 📬 Contact

Feel free to open issues or reach out via email if you use the simulator or need help adapting it to your own targets.

