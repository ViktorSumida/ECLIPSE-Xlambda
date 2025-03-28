from IPython.display import HTML
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from star import Star
from Planeta import Planeta
from verify import calculaLat
from keplerAux import keplerfunc  # biblioteca auxiliar caso a kepler não funcione
import matplotlib.animation as animation
import os
import platform
import time
from numba import njit, prange

# Função acelerada com Numba para o cálculo da curva de luz
@njit(parallel=True)
def curvaLuz_jit(x0, y0, tamanhoMatriz, raioPlanetaPixel, estrelaManchada_flat, maxCurvaLuz):
    """
    Calcula a contribuição da estrela para a curva de luz,
    somando a intensidade de cada pixel que NÃO está coberto pelo planeta.
    
    Parâmetros:
      - x0, y0: coordenadas do centro do planeta na matriz.
      - tamanhoMatriz: dimensão (N) da matriz (assume-se quadrada).
      - raioPlanetaPixel: raio do planeta em pixels.
      - estrelaManchada_flat: vetor (flatten) da matriz da estrela.
      - maxCurvaLuz: soma total das intensidades da estrela para normalização.
      
    Retorna:
      - Valor normalizado da curva de luz para a posição (x0, y0).
    """
    N = tamanhoMatriz
    total = 0.0
    for i in prange(N * N):
        row = i / N
        col = i - N * math.floor(i / N)
        if ((row - y0)**2 + (col - x0)**2) > (raioPlanetaPixel**2):
            total += estrelaManchada_flat[i]
    return total / maxCurvaLuz

# Classe Eclipse (versão sem CME e sem luas)
class Eclipse:
    def __init__(self, Nx, Ny, raio_estrela_pixel, estrela_manchada: Star, planeta_: Planeta):
        """
        Parâmetros:
          - Nx e Ny: dimensões da matriz da estrela.
          - raio_estrela_pixel: raio da estrela em pixels.
          - estrela_manchada: objeto Estrela (já com a matriz gerada, possivelmente com manchas).
          - planeta_: objeto Planeta com os parâmetros orbitais.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.tamanhoMatriz = self.Nx  # assume matriz quadrada
        
        # Estrela
        self.raio_estrela_pixel = raio_estrela_pixel
        self.estrela_ = estrela_manchada
        self.estrela_matriz = estrela_manchada.getMatrizEstrela()
        
        # Planeta
        self.planeta_ = planeta_
        
        # OUTPUT: inicializa a curva de luz com 1.0 em cada posição
        self.curvaLuz = [1.0 for i in range(self.Nx)]
    
    def setTempoHoras(self, intervalo):
        """
        Define o vetor de tempo (em horas) para o trânsito com base no intervalo em minutos.
        """
        self.intervaloTempo = intervalo
        self.tamanhoMatriz = self.Nx
        self.tempoHoras = (np.arange(self.tamanhoMatriz) - self.tamanhoMatriz / 2) * self.intervaloTempo / 60.

    def criarEclipse(self, plot_anim, plot_graph):
        """
        Generates the eclipse model and computes the transit light curve,
        automatically estimating the time interval and number of points.
        """
        self.plot_anim = plot_anim
        self.plot_graph = plot_graph

        # Automatic calculation of interval and number of points
        N_points, self.intervaloTempo, self.tempoTotal = self.calculateAutomaticNPoints()
        
        # Create the time vector
        self.setTempoHoras(self.intervaloTempo)
        
        tamanhoMatriz = self.tamanhoMatriz
        dtor = np.pi / 180.
        semiEixoPixel = self.planeta_.semiEixoRaioStar * self.raio_estrela_pixel

        # Parâmetros orbitais do planeta
        nk = 2 * np.pi / (self.planeta_.periodo * 24)  # em horas⁻¹
        Tp = self.planeta_.periodo * self.planeta_.anom / 360. * 24.  # tempo do pericentro (horas)
        m = nk * (self.tempoHoras - Tp)  # em radianos

        # Cálculo da anomalia excentrica (usando a função keplerfunc)s
        eccanom = keplerfunc(m, self.planeta_.ecc)
        xs = semiEixoPixel * (np.cos(eccanom) - self.planeta_.ecc)
        ys = semiEixoPixel * (np.sqrt(1 - self.planeta_.ecc**2) * np.sin(eccanom))

        ang = self.planeta_.anom * dtor - (np.pi / 2)
        xp = xs * np.cos(ang) - ys * np.sin(ang)
        yp = xs * np.sin(ang) + ys * np.cos(ang)

        # Ajusta as posições em relação ao centro da matriz
        ie = np.where(np.abs(self.tempoHoras) == np.min(np.abs(self.tempoHoras)))[0]
        xplaneta = xp - xp[ie[0]]
        yplaneta = yp * np.cos(self.planeta_.anguloInclinacao * dtor)

        # Seleciona os pontos de interesse para o trânsito
        pp = np.where((np.abs(xplaneta) < 1.2 * tamanhoMatriz / 2) &
                    (np.abs(yplaneta) < tamanhoMatriz / 2))[0]
        xplan = xplaneta[pp] + tamanhoMatriz / 2
        yplan = yplaneta[pp] + tamanhoMatriz / 2

        # Define o raio do planeta em pixels e o valor máximo para normalização
        raioPlanetaPixel = float(self.planeta_.raioPlanetaRstar * self.raio_estrela_pixel)
        maxCurvaLuz = np.sum(self.estrela_matriz)

        # Calcula a duração do trânsito
        # (latitudeTransito é calculada com base na semi-eixo do planeta)
        
        #latitudeTransito = -np.arcsin(self.planeta_.semiEixoRaioStar * np.cos(self.planeta_.anguloInclinacao * dtor)) / dtor
        #duracaoTransito = 2 * (90. - np.arccos((np.cos(latitudeTransito * dtor)) / self.planeta_.semiEixoRaioStar) / dtor) * self.planeta_.periodo / 360 * 24.
        #tempoTotal = 3 * duracaoTransito
        #self.tempoTotal = tempoTotal

        # Cálculo do número de pontos na curva de luz
        nn = np.fix(self.tempoTotal * 60. / self.intervaloTempo)

        # Para processamento, converte a matriz da estrela em vetor
        estrela_matriz_flat = self.estrela_matriz.flatten().astype(np.float64)

        # Atualiza a curva de luz para os pontos de interesse usando a função JIT acelerada
        for i in range(len(pp)):
            x0 = xplan[i]
            y0 = yplan[i]
            self.curvaLuz[pp[i]] = curvaLuz_jit(x0, y0, tamanhoMatriz, raioPlanetaPixel,
                                                estrela_matriz_flat, maxCurvaLuz)

        if plot_anim == True:
            # Exemplo simples de animação (pode ser adaptado conforme necessário)
            fig, ax1 = plt.subplots()
            ims = []
            numAux = 0
            plota = True
            for i in range(len(pp)):
                x0 = xplan[i]
                y0 = yplan[i]
                if plota and self.curvaLuz[pp[i]] != 1 and numAux < 200:
                    plan = np.ones(tamanhoMatriz * tamanhoMatriz, dtype=float)
                    kk = np.arange(tamanhoMatriz * tamanhoMatriz, dtype=np.float64)
                    row_coord = kk / tamanhoMatriz
                    col_coord = kk - tamanhoMatriz * np.floor(kk / tamanhoMatriz)
                    mask = ((row_coord - y0)**2 + (col_coord - x0)**2 <= raioPlanetaPixel**2)
                    plan[mask] = 0.
                    plan = plan.reshape(tamanhoMatriz, tamanhoMatriz)
                    plt.axis([0, self.Nx, 0, self.Ny])
                    im = ax1.imshow(self.estrela_matriz * plan, cmap="hot", animated=True)
                    ims.append([im])
                    numAux += 1
                plota = not(plota)
            from matplotlib.animation import FuncAnimation
            def update(frame):
                im = ims[frame][0]
                ax1.clear()
                ax1.imshow(im.get_array(), cmap="hot", animated=True)
                ax1.set_title('')
                return [im]
            ani = FuncAnimation(fig, update, frames=len(ims), blit=True)
            plt.ion()
            plt.show(block=True)
        else:
            if plot_graph == True:
                plt.figure(figsize=(10, 5))
                plt.plot(self.tempoHoras, self.curvaLuz)
                plt.xlabel('Time from Transit Center [h]')
                plt.ylabel('Relative Flux')
                plt.title('Transit Light Curve')
                plt.show()

    def calculaLatMancha(self):
        latsugerida = calculaLat(self.planeta_.semiEixoRaioStar, self.planeta_.anguloInclinacao)
        print("A latitude sugerida para que a mancha influencie na curva de luz da estrela é:", latsugerida)
        return latsugerida

    def calculaLongMancha(self, a, time, lat):
        latitude_rad = math.radians(lat)
        angle = math.radians(90) - (math.radians(360) * time) / (24 * self.planeta_.periodo)
        longitude = math.degrees(math.asin((a * math.cos(angle)) / math.cos(abs(latitude_rad))))
        #print("A longitude sugerida para que a mancha influencie na curva de luz da estrela é:", longitude)
        return longitude

    def getMatrizTransformada(self, estrela):
        if isinstance(estrela, Star):
            estrela = estrela.getMatrizEstrela()
        if not isinstance(estrela, np.ndarray):
            raise TypeError("O parâmetro 'estrela' deve ser uma matriz NumPy.")
        
        # Apenas retorna a matriz NumPy sem conversão para ponteiro
        return np.array(estrela, dtype=np.float64)

    def getTempoTransito(self):
        return self.tempoTotal

    def getTempoHoras(self):
        return self.tempoHoras

    def getCurvaLuz(self):
        return self.curvaLuz

    def setEstrela(self, estrela):
        self.estrela_matriz = estrela

    def getError(self):
        return self.error
    
    def calculateAutomaticNPoints(
        self,
        percent_resolution=0.01, # based on the level of detail (resolution) you want
        min_interval_limit=0.1,
        max_interval_limit=10.0
    ):
        """
        Calculates the number of points and interval for the light curve,
        using adaptive resolution based on transit duration.
        
        Parameters:
            - percent_resolution: fraction of the transit duration per point (default: 1% = 0.01)
            - min_interval_limit: minimum allowed interval (in minutes, default: 0.1)
            - max_interval_limit: maximum allowed interval (in minutes, default: 10.0)

        Returns:
            - N_points: number of points for the light curve
            - interval_minutes: interval between points (in minutes)
            - total_time: total observation time (in hours)
        """
        dtor = np.pi / 180.

        a_rs = self.planeta_.semiEixoRaioStar
        inc = self.planeta_.anguloInclinacao
        periodo = self.planeta_.periodo

        # Estimate transit latitude and duration (in hours)
        latitude_transit = -np.arcsin(a_rs * np.cos(inc * dtor)) / dtor
        transit_duration = 2 * (90. - np.arccos(np.cos(latitude_transit * dtor) / a_rs) / dtor) * periodo / 360 * 24.

        total_time = 3 * transit_duration  # in hours
        self.tempoTotal = total_time

        # Adaptive interval in minutes, based on % of transit duration
        interval_adaptive = (transit_duration * 60) * percent_resolution

        # Clamp to user-defined limits (min_interval_limit ≤ interval_minutes ≤ max_interval_limit)
        interval_minutes = max(min_interval_limit, min(interval_adaptive, max_interval_limit))

        N_points = int(np.ceil((total_time * 60) / interval_minutes))

        return N_points, interval_minutes, total_time


