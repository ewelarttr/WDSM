import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GeneratorRozkladow:
    def __init__(self, ziarno=None):
        self.ziarno = ziarno
        if ziarno is not None:
            np.random.seed(ziarno)

    def generator_poissona(self, lambda_param, n):
        if lambda_param <= 0:
            raise ValueError("Parametr λ musi być dodatni")

        wyniki = np.zeros(n, dtype=int)

        for i in range(n):
            u = np.random.random()
            k = 0
            p = np.exp(-lambda_param)
            F = p

            while u > F:
                k += 1
                p *= lambda_param / k
                F += p

            wyniki[i] = k

        return wyniki

    def generator_normalny(self, mu, sigma, n):
        if sigma <= 0:
            raise ValueError("Odchylenie standardowe musi być dodatnie")

        m = n if n % 2 == 0 else n + 1
        wyniki = np.zeros(m)

        for i in range(0, m, 2):
            u1 = np.random.random()
            u2 = np.random.random()

            R = np.sqrt(-2 * np.log(u1))
            theta = 2 * np.pi * u2

            z1 = R * np.cos(theta)
            z2 = R * np.sin(theta)

            wyniki[i] = mu + sigma * z1
            if i + 1 < m:
                wyniki[i + 1] = mu + sigma * z2

        return wyniki[:n]

    def rysuj_histogram(self, dane, tytul, rozklad_teoretyczny=None, parametry=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        if rozklad_teoretyczny == 'poisson':
            unique, counts = np.unique(dane, return_counts=True)
            ax.bar(unique, counts / len(dane), alpha=0.7, label='Empiryczny')

            if parametry:
                lambda_param = parametry[0]
                x_teor = np.arange(0, max(dane) + 2)
                y_teor = stats.poisson.pmf(x_teor, lambda_param)
                ax.plot(x_teor, y_teor, 'r-', linewidth=2, label='Teoretyczny')

        elif rozklad_teoretyczny == 'normal':
            ax.hist(dane, bins=30, density=True, alpha=0.7, label='Empiryczny')

            if parametry:
                mu, sigma = parametry
                x_teor = np.linspace(min(dane), max(dane), 100)
                y_teor = stats.norm.pdf(x_teor, mu, sigma)
                ax.plot(x_teor, y_teor, 'r-', linewidth=2, label='Teoretyczny')

        ax.set_xlabel('Wartość')
        ax.set_ylabel('Prawdopodobieństwo')
        ax.set_title(tytul)
        ax.legend()
        ax.grid(True, alpha=0.3)

        statystyki = f'Średnia: {np.mean(dane):.3f}\nOdch. std: {np.std(dane):.3f}'
        if rozklad_teoretyczny == 'poisson' and parametry:
            statystyki += f'\nλ teoret: {parametry[0]:.3f}'
        elif rozklad_teoretyczny == 'normal' and parametry:
            statystyki += f'\nμ teoret: {parametry[0]:.3f}\nσ teoret: {parametry[1]:.3f}'

        ax.text(0.02, 0.98, statystyki, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig


class AplikacjaGenerujaca:
    def __init__(self, root):
        self.root = root
        self.root.title("Generator Rozkładów Dyskretnych")
        self.root.geometry("1200x700")

        self.generator = None
        self.aktualne_dane = None

        self._tworz_interfejs()

    def _tworz_interfejs(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        left_frame = ttk.LabelFrame(main_frame, text="Parametry generatora", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        ttk.Label(left_frame, text="Ziarno (opcjonalnie):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ziarno_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.ziarno_var, width=20).grid(row=0, column=1, pady=5)

        ttk.Label(left_frame, text="Liczba próbek:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.n_prob_var = tk.StringVar(value="10000")
        ttk.Entry(left_frame, textvariable=self.n_prob_var, width=20).grid(row=1, column=1, pady=5)

        ttk.Separator(left_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(left_frame, text="ROZKŁAD POISSONA", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2,
                                                                                        pady=5)

        ttk.Label(left_frame, text="λ (lambda):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.lambda_var = tk.StringVar(value="5.0")
        ttk.Entry(left_frame, textvariable=self.lambda_var, width=20).grid(row=4, column=1, pady=5)

        ttk.Button(left_frame, text="Generuj Poisson",
                   command=self.generuj_poissona).grid(row=5, column=0, columnspan=2, pady=10)

        ttk.Separator(left_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(left_frame, text="ROZKŁAD NORMALNY", font=('Arial', 10, 'bold')).grid(row=7, column=0, columnspan=2,
                                                                                        pady=5)

        ttk.Label(left_frame, text="μ (średnia):").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.mu_var = tk.StringVar(value="0.0")
        ttk.Entry(left_frame, textvariable=self.mu_var, width=20).grid(row=8, column=1, pady=5)

        ttk.Label(left_frame, text="σ (odch. std):").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.sigma_var = tk.StringVar(value="1.0")
        ttk.Entry(left_frame, textvariable=self.sigma_var, width=20).grid(row=9, column=1, pady=5)

        ttk.Button(left_frame, text="Generuj Normalny",
                   command=self.generuj_normalny).grid(row=10, column=0, columnspan=2, pady=10)

        ttk.Button(left_frame, text="Resetuj ziarno",
                   command=self.resetuj_ziarno).grid(row=11, column=0, columnspan=2, pady=20)

        right_frame = ttk.LabelFrame(main_frame, text="Wizualizacja rozkładu", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.fig_canvas = None
        self.wykres_frame = ttk.Frame(right_frame)
        self.wykres_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

    def _pobierz_ziarno(self):
        try:
            return int(self.ziarno_var.get()) if self.ziarno_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Błąd", "Ziarno musi być liczbą całkowitą")
            return None

    def _pobierz_n_prob(self):
        try:
            n = int(self.n_prob_var.get())
            if n <= 0:
                raise ValueError
            return n
        except ValueError:
            messagebox.showerror("Błąd", "Liczba próbek musi być dodatnią liczbą całkowitą")
            return 1000

    def resetuj_ziarno(self):
        self.ziarno_var.set("")
        self.generator = GeneratorRozkladow()
        messagebox.showinfo("Info", "Ziarno zostało zresetowane")

    def generuj_poissona(self):
        try:
            ziarno = self._pobierz_ziarno()
            n = self._pobierz_n_prob()
            lambda_param = float(self.lambda_var.get())

            if lambda_param <= 0:
                raise ValueError("λ musi być dodatnie")

            self.generator = GeneratorRozkladow(ziarno)
            self.aktualne_dane = self.generator.generator_poissona(lambda_param, n)

            fig = self.generator.rysuj_histogram(
                self.aktualne_dane,
                f"Rozkład Poissona (λ={lambda_param}, n={n})",
                rozklad_teoretyczny='poisson',
                parametry=(lambda_param,)
            )

            self._wyswietl_wykres(fig)

        except ValueError as e:
            messagebox.showerror("Błąd", str(e))

    def generuj_normalny(self):
        try:
            ziarno = self._pobierz_ziarno()
            n = self._pobierz_n_prob()
            mu = float(self.mu_var.get())
            sigma = float(self.sigma_var.get())

            if sigma <= 0:
                raise ValueError("Odchylenie standardowe musi być dodatnie")

            self.generator = GeneratorRozkladow(ziarno)
            self.aktualne_dane = self.generator.generator_normalny(mu, sigma, n)

            fig = self.generator.rysuj_histogram(
                self.aktualne_dane,
                f"Rozkład Normalny (μ={mu}, σ={sigma}, n={n})",
                rozklad_teoretyczny='normal',
                parametry=(mu, sigma)
            )

            self._wyswietl_wykres(fig)

        except ValueError as e:
            messagebox.showerror("Błąd", str(e))

    def _wyswietl_wykres(self, fig):
        for widget in self.wykres_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.wykres_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = AplikacjaGenerujaca(root)
    root.mainloop()


if __name__ == "__main__":
    main()
