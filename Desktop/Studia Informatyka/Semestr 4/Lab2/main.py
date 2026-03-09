import os
import subprocess
import sys



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import pandas as pd
    import xlrd
    import matplotlib.pyplot as plt
except ImportError:
    install('pandas')
    install('xlrd==2.0.1')
    install('openpyxl')
    install('matplotlib')
    import pandas as pd
    import matplotlib.pyplot as plt

import numpy as np
import math

# --- KONFIGURACJA ---
PLIK_WEJSCIOWY = 'WdSM_lab02_analiza_systemu.xls'
PLIK_WYJSCIOWY = 'wyniki_zadanie5.xlsx'
R = 2  # Liczba stanowisk z zadania 4


def oblicz_p0(lm, r, rho):
    if rho >= 1: return 0
    suma = sum([(lm ** k) / math.factorial(k) for k in range(r)])
    ogon = (lm ** r) / (math.factorial(r) * (1 - rho))
    return 1 / (suma + ogon)


if not os.path.exists(PLIK_WEJSCIOWY):
    print(f"!!! BRAK PLIKU: Upewnij się, że plik nazywa się {PLIK_WEJSCIOWY} i jest w tym samym folderze !!!")
else:
    try:

        df = pd.read_excel(PLIK_WEJSCIOWY, engine='xlrd')


        czasy_p = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        czasy_o = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        dane = pd.DataFrame({'przybycia': czasy_p, 'obsluga': czasy_o}).dropna().values

        wyniki = []

        for i in range(1, len(dane) + 1):
            subset = dane[:i]
            sr_p = np.mean(subset[:, 0])
            sr_o = np.mean(subset[:, 1])


            lam = 60 / sr_p if sr_p > 0 else 0
            mu = 60 / sr_o if sr_o > 0 else 0
            rho = lam / (R * mu) if (R * mu) > 0 else 0

            q, w, p0 = 0, 0, 0
            if rho < 1:
                lm_ratio = lam / mu
                p0 = oblicz_p0(lm_ratio, R, rho)
                q = (p0 * (lm_ratio ** R) * rho) / (math.factorial(R) * (1 - rho) ** 2)
                w = q / lam if lam > 0 else 0
            else:
                q, w = np.nan, np.nan

            wyniki.append({'Lp': i, 'Lambda': lam, 'Mu': mu, 'Rho': rho, 'P0': p0, 'Q': q, 'W': w})


        df_final = pd.DataFrame(wyniki)
        df_final.to_excel(PLIK_WYJSCIOWY, index=False)
        print(f"--- SUKCES: Dane zapisane w {PLIK_WYJSCIOWY} ---")


        plt.figure(figsize=(10, 12))

        plt.subplot(3, 1, 1)
        plt.plot(df_final['Lambda'], label='Lambda (λ)')
        plt.plot(df_final['Mu'], label='Mu (μ)')
        plt.title('Stopa przybyć i obsługi')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(df_final['Rho'], color='red', label='Intensywność (ρ)')
        plt.axhline(y=1, color='black', linestyle='--')
        plt.title('Intensywność ruchu')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df_final['Q'], label='Kolejka (Q)')
        plt.plot(df_final['W'], label='Czas (W)')
        plt.title('Kolejka i czas oczekiwania')
        plt.legend()

        plt.tight_layout()
        plt.savefig('wykresy_zadanie5.png')
        print("--- SUKCES: Wykresy zapisane jako 'wykresy_zadanie5.png' ---")
        plt.show()

    except Exception as e:
        print(f"BŁĄD: {e}")