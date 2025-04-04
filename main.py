import numpy as np
import os
from matplotlib import pyplot
from star import Star
from eclipse_nv1 import Eclipse
from Planeta import Planeta

def planck(wavelength_m, Temp):
    """
    Planck function in SI units.
    wavelength_m: float (in meters)
    Temp: float (in Kelvin)
    """
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    aux = (h * c) / (wavelength_m * k * Temp)
    return (2 * h * c**2) / (wavelength_m**5 * (np.exp(aux) - 1.0))

BASELINE_WAVE = None
BASELINE_DLAMDA = None

class MainProgram:
    def __init__(
        self,
        *,
        target,
        num_elements,
        profile,
        c1, c2, c3, c4,
        lambdaEff,
        intensidadeMaxima,
        raioStar,
        ecc, anom,
        tempStar,
        starspots,
        quantidade,
        lat, longt,
        r=None,
        r_spot=None,
        r_facula=None,
        semiEixoUA,
        massStar,
        plot_anim,
        periodo,
        anguloInclinacao,
        raioPlanetaRj,
        plot_graph,
        plot_star,
        tempSpot,
        tempFacula,
        fillingFactor,
        min_pixels,
        max_pixels,
        pixels_per_rp,
        simulation_mode="unspotted",
        both_mode=False
    ):
        self.target           = target
        self.num_elements     = num_elements
        self.profile          = profile
        self.c1               = c1
        self.c2               = c2
        self.c3               = c3
        self.c4               = c4
        self.lambdaEff        = lambdaEff
        self.intensidadeMaxima= intensidadeMaxima
        self.raioStar         = raioStar
        self.ecc              = ecc
        self.anom             = anom
        self.tempStar         = tempStar
        self.starspots        = starspots
        self.quantidade       = quantidade
        self.lat              = lat
        self.longt            = longt
        self.r                = r  # used in spot/faculae
        self.r_spot           = r_spot if r_spot is not None else r  # fallback to r
        self.r_facula         = r_facula if r_facula is not None else r  # fallback to r
        self.semiEixoUA       = semiEixoUA
        self.massStar         = massStar
        self.plot_anim        = plot_anim
        self.periodo          = periodo
        self.anguloInclinacao = anguloInclinacao
        self.raioPlanetaRj    = raioPlanetaRj
        self.plot_graph       = plot_graph
        self.plot_star        = plot_star
        self.tempSpot         = tempSpot
        self.tempFacula       = tempFacula
        self.fillingFactor    = fillingFactor
        self.simulation_mode  = simulation_mode
        self.min_pixels       = min_pixels
        self.max_pixels       = max_pixels
        self.pixels_per_rp    = pixels_per_rp
        self.both_mode        = both_mode
        
        # Adjust r_spot and r_facula for individual modes
        if not both_mode:
            if simulation_mode == "spot" and self.fillingFactor[0] > 0 and self.quantidade > 0:
                self.r_spot = np.sqrt(self.fillingFactor[0] / self.quantidade)
            elif simulation_mode == "faculae" and self.fillingFactor[0] > 0 and self.quantidade > 0:
                self.r_facula = np.sqrt(self.fillingFactor[0] / self.quantidade)

        # Adjust r_spot and r_facula from r only if both_mode and no r_spot/r_facula given
        if both_mode and r_spot is None and r_facula is None and self.r is not None:
            self.r_facula = self.r * np.sqrt(self.quantidade)
            self.r_spot = self.r * np.sqrt(self.quantidade)
   
        # Convert star radius from solar radii to km
        raioStar_km = self.raioStar * 696340

        # Create the planet
        self.planeta_ = Planeta(
            self.semiEixoUA, self.raioPlanetaRj,
            self.periodo, self.anguloInclinacao,
            self.ecc, self.anom,
            raioStar_km, 0
        )

        (self.tamanhoMatriz, _, self.star_pixels) = self.calculateMatrixFromTransit(
            self.planeta_, self.min_pixels, self.max_pixels, self.pixels_per_rp
        )

        self.run_simulations()

    def run_simulations(self):
        # Arrays to store final results
        stack_curva = None
        stack_tempo = None
        D_lambda    = np.zeros(self.num_elements)
        lambdaEff_nm= np.zeros(self.num_elements)

        # Reference planck at star's peak
        peak_wavelength = 2.8976e-3 / self.tempStar
        star_peak_val   = planck(peak_wavelength, self.tempStar)

        for i in range(self.num_elements):
            lam_m    = self.lambdaEff[i] * 1.0e-6
            star_val = planck(lam_m, self.tempStar)
            # Normalized star intensity at this wavelength
            star_intensity_norm = (self.intensidadeMaxima * star_val) / star_peak_val

            # Create star object
            estrela_ = Star(
                self.star_pixels,
                self.raioStar,
                star_intensity_norm,
                self.c1[i], self.c2[i],
                self.c3[i], self.c4[i],
                self.tamanhoMatriz
            )

            if (not self.starspots) or (self.simulation_mode == "unspotted"):
                matriz_estrela   = estrela_.getMatrizEstrela()
                intensity_active = 1.0

            elif self.simulation_mode == "spot":
                if self.tempSpot is not None and self.fillingFactor[0] > 0:
                    intensity_spot = planck(lam_m, self.tempSpot)
                    ratio_spot     = (intensity_spot * star_intensity_norm) / star_val / star_intensity_norm

                    if self.simulation_mode == "spot" and self.both_mode==False:
                        # both_mode=False, usa múltiplas manchas
                        for j in range(self.quantidade):
                            estrela_.addMancha(Star.Mancha(ratio_spot, self.r_spot, self.lat[j], self.longt[j]))
                    else:
                        # both_mode=True, usa índice fixo da mancha
                        estrela_.addMancha(Star.Mancha(ratio_spot, self.r_spot, self.lat[1], self.longt[1]))

                    estrela_.criaEstrelaManchada()
                    matriz_estrela   = estrela_.getMatrizEstrela()
                    intensity_active = ratio_spot
                else:
                    print("Warning: tempSpot is None or filling factor is 0. No spot added.")
                    matriz_estrela   = estrela_.getMatrizEstrela()
                    intensity_active = 1.0

            elif self.simulation_mode == "faculae":
                if self.tempFacula is None:
                    raise ValueError("tempFacula must be provided if simulation_mode='faculae'!")

                intensity_fac = planck(lam_m, self.tempFacula)
                ratio_fac     = (intensity_fac * star_intensity_norm) / star_val / star_intensity_norm

                if self.simulation_mode == "faculae" and self.both_mode==False:
                    # both_mode=False, usa múltiplas fáculas
                    for j in range(self.quantidade):
                        estrela_.addMancha(Star.Mancha(ratio_fac, self.r_facula, self.lat[j], self.longt[j]))
                else:
                    # both_mode=True, usa índice fixo da facula
                    estrela_.addMancha(Star.Mancha(ratio_fac, self.r_facula, self.lat[0], self.longt[0]))

                estrela_.criaEstrelaManchada()
                matriz_estrela   = estrela_.getMatrizEstrela()
                intensity_active = ratio_fac
                

            elif self.simulation_mode == "both":

                f_facula = self.fillingFactor[0] if len(self.fillingFactor) > 0 else 0.0
                f_spot   = self.fillingFactor[1] if len(self.fillingFactor) > 1 else 0.0

                has_facula = f_facula > 0
                has_spot   = f_spot > 0

                added_facula = False
                added_spot   = False

                if has_facula and has_spot:
                    intensity_fac = planck(lam_m, self.tempFacula)
                    ratio_fac = (intensity_fac * star_intensity_norm) / star_val / star_intensity_norm
                    estrela_.addFacula(Star.Facula(self.r_facula, ratio_fac, self.lat[0], self.longt[0]))
                    added_facula = True

                    intensity_spot = planck(lam_m, self.tempSpot)
                    ratio_spot = (intensity_spot * star_intensity_norm) / star_val / star_intensity_norm
                    estrela_.addMancha(Star.Mancha(ratio_spot, self.r_spot, self.lat[1], self.longt[1]))
                    added_spot = True

                elif has_spot:
                    index_spot = 1 if len(self.lat) > 1 else 0
                    intensity_spot = planck(lam_m, self.tempSpot)
                    ratio_spot = (intensity_spot * star_intensity_norm) / star_val / star_intensity_norm
                    estrela_.addMancha(Star.Mancha(ratio_spot, self.r_spot, self.lat[index_spot], self.longt[index_spot]))
                    added_spot = True

                elif has_facula:
                    intensity_fac = planck(lam_m, self.tempFacula)
                    ratio_fac = (intensity_fac * star_intensity_norm) / star_val / star_intensity_norm
                    estrela_.addFacula(Star.Facula(self.r_facula, ratio_fac, self.lat[0], self.longt[0]))
                    added_facula = True

                if added_spot:
                    estrela_.criaEstrelaManchada()
                if added_facula:
                    estrela_.criaEstrelaComFaculas()

                matriz_estrela = estrela_.getMatrizEstrela()

                # Decide qual intensidade ativa salvar
                if added_spot:
                    intensity_active = ratio_spot
                elif added_facula:
                    intensity_active = ratio_fac
                else:
                    intensity_active = 1.0

            # Optionally plot star
            if self.plot_star:
                estrela_.Plotar(self.tamanhoMatriz, matriz_estrela)

            # Run eclipse
            eclipse_ = Eclipse(self.tamanhoMatriz, self.tamanhoMatriz, self.star_pixels, estrela_, self.planeta_)
            eclipse_.setEstrela(matriz_estrela)
            eclipse_.criarEclipse(self.plot_anim, self.plot_graph)
            eclipse_.setTempoHoras(eclipse_.intervaloTempo)

            curva = eclipse_.getCurvaLuz()
            tempo = eclipse_.getTempoHoras()
            tempoTransito = eclipse_.getTempoTransito()

            if stack_curva is None or stack_curva.shape[1] == 0:
                stack_curva = np.zeros((self.num_elements, len(curva)))
                stack_tempo = np.zeros((self.num_elements, len(tempo)))

            stack_curva[i, :] = curva
            stack_tempo[i, :] = tempo

            idx_mid = len(curva)//2
            D_lambda[i] = (1.0 - curva[idx_mid]) * 1e6
            lambdaEff_nm[i] = self.lambdaEff[i]*1000

        # Finally plot the results
        self.plot_curves(stack_tempo, stack_curva, lambdaEff_nm, D_lambda, tempoTransito)
        
        # Extract safe values for saving
        f_spot = self.fillingFactor[0] if len(self.fillingFactor) > 0 else 0.0
        f_facula = self.fillingFactor[1] if len(self.fillingFactor) > 1 else np.nan

        self.salvar_dados_simulacao(
            f_spot=f_spot,
            tempSpot=self.tempSpot,
            f_facula=f_facula,
            tempFacula=self.tempFacula,
            lambdaEff_nm=lambdaEff_nm,
            D_lambda=D_lambda
        )

    def plot_curves(self, stack_tempo, stack_curva, lambdaEff_nm, D_lambda, tempoTransito):
        palette = pyplot.cm.cool_r(np.linspace(0,1,self.num_elements))
        #tempoTransito = (max(stack_tempo[0]) - min(stack_tempo[0])) / 2
        min_flux = stack_curva.min()

        cp = self.num_elements - 1
        for i in range(self.num_elements):
            pyplot.plot(stack_tempo[i], stack_curva[i],
                        label=f"{lambdaEff_nm[i]:.0f} nm",
                        color=palette[cp])
            cp -= 1

        pyplot.axis([-tempoTransito/5, tempoTransito/5, min_flux, 1.00005])
        pyplot.xlabel("Time from transit center (hr)")
        pyplot.ylabel("Relative flux")
        #pyplot.legend()
        pyplot.tight_layout()
        pyplot.show()

    def calculateMatrixFromTransit(self, planet, min_pixels, max_pixels, pixels_per_rp, margin=0.1):
        """
        Calculates the matrix size based on the stellar radius and desired resolution.

        Parameters:
            - planet: Planeta object (must contain raioPlanetaRstar)
            - min_pixels: minimum allowed matrix size
            - max_pixels: maximum allowed matrix size
            - pixels_per_rp: desired pixels per planetary radius (e.g., 50)
            - margin: extra space beyond the stellar disk (e.g., 0.1 for 10%)

        Returns:
            - Nx, Ny: matrix dimensions (square)
            - star_radius_pixels: stellar radius in pixels
        """
        rp_rstar = planet.raioPlanetaRstar

        # Calculate how many pixels per R★ are needed to achieve the desired pixels per Rp
        pixels_per_rstar = pixels_per_rp / rp_rstar

        # Clamp if necessary (not mandatory here, but for safety)
        pixels_per_rstar = np.clip(pixels_per_rstar, min_pixels / 2, max_pixels / 2)

        # Total matrix covers 2*R★ + margin
        matrix_size = int(2 * pixels_per_rstar * (1 + margin))
        matrix_size = int(np.clip(matrix_size, min_pixels, max_pixels))

        return matrix_size, matrix_size, pixels_per_rstar
    
    
    def salvar_dados_simulacao(
        self,
        f_spot,
        tempSpot,
        lambdaEff_nm,
        D_lambda,
        f_facula=None,
        tempFacula=None
    ):
        """
        Saves the simulation data. Always creates 6 columns:
        f_spot, tempSpot, f_facula, tempFacula, wavelength [nm], D_lambda [ppm]
        
        In both_mode==True, the simulation_mode distinguishes between:
        - "both": both spot and facula values are used.
        - "spot": only spot values are used (facula values are set to NaN).
        - "faculae": only facula values are used (spot values are set to NaN).
        
        In both_mode==False, the filling factors are computed as:
        f_total = quantidade * (r**2)
        """
        out_file = f"simulation_results_{self.target}.txt"
        file_exists = os.path.exists(out_file)

        # Compute filling factors and adjust temperatures consistently
        if self.both_mode:
            if self.simulation_mode == "both":
                f_spot_total   = self.r_spot**2 if self.r_spot is not None else np.nan
                f_facula_total = self.r_facula**2 if self.r_facula is not None else np.nan
                tempSpot   = tempSpot if self.r_spot and self.r_spot > 0 else np.nan
                tempFacula = tempFacula if self.r_facula and self.r_facula > 0 else np.nan
                if f_spot_total == 0.0:
                    tempSpot = np.nan
                if f_facula_total == 0.0:
                    tempFacula = np.nan

            elif self.simulation_mode == "spot":
                # In a spot-only run, use only spot values; facula values become NaN.
                f_spot_total   = self.r_spot**2 if self.r_spot is not None else np.nan
                f_facula_total = np.nan
                tempSpot   = tempSpot if self.r_spot and self.r_spot > 0 else np.nan
                tempFacula = np.nan
                if f_spot_total == 0.0:
                    tempSpot = np.nan

            elif self.simulation_mode == "faculae":
                # In a facula-only run, use only facula values; spot values become NaN.
                f_facula_total = self.r_facula**2 if self.r_facula is not None else np.nan
                f_spot_total   = np.nan
                tempFacula = tempFacula if self.r_facula and self.r_facula > 0 else np.nan
                tempSpot   = np.nan
                if f_facula_total == 0.0:
                    tempFacula = np.nan
            else:
                # Fallback: if simulation_mode is undefined.
                f_spot_total = np.nan
                f_facula_total = np.nan
                tempSpot = np.nan
                tempFacula = np.nan

        else:
            # both_mode == False: the total filling factor is computed using 'quantidade'
            if self.simulation_mode == "spot":
                f_spot_total   = self.quantidade * self.r_spot**2 if self.r_spot is not None and self.quantidade > 0 else np.nan
                f_facula_total = np.nan
                tempSpot   = tempSpot if f_spot_total and f_spot_total > 0 else np.nan
                tempFacula = np.nan
                if f_spot_total == 0.0:
                    tempSpot = np.nan

            elif self.simulation_mode == "faculae":
                f_facula_total = self.quantidade * self.r_facula**2 if self.r_facula is not None and self.quantidade > 0 else np.nan
                f_spot_total   = np.nan
                tempFacula = tempFacula if f_facula_total and f_facula_total > 0 else np.nan
                tempSpot   = np.nan
                if f_facula_total == 0.0:
                    tempFacula = np.nan

            else:
                # Spotless run
                f_spot_total   = 0.0
                f_facula_total = np.nan
                tempSpot = np.nan
                tempFacula = np.nan

        # Prepare data array for all wavelengths
        n = len(lambdaEff_nm)
        data = np.column_stack([
            np.full(n, f_spot_total, dtype=float),
            np.full(n, tempSpot, dtype=float),
            np.full(n, f_facula_total, dtype=float),
            np.full(n, tempFacula, dtype=float),
            np.array(lambdaEff_nm, dtype=float),
            np.array(D_lambda, dtype=float)
        ])

        # Write (append) the data to file
        with open(out_file, 'a') as f:
            if not file_exists:
                f.write("f_spot,tempSpot,f_facula,tempFacula,wavelength,D_lambda\n")
            np.savetxt(f, data, delimiter=",", fmt="%.6f")

