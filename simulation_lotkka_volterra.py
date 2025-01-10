import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

# Chargement des données réelles
df = pd.read_csv("populations_lapins_renards.csv")
df_cp = df.copy()
df_cp["days"] = range(1, 1001)  # Ajouter une colonne pour les jours (1 à 1000)

# Paramètres du modèle
#alpha = 1 / 3  # Taux de reproduction des lapins
#beta = 1 / 3   # Taux de mortalité des lapins par les prédateurs
#delta = 1      # Taux de reproduction des renards en fonction des proies
#gamma = 1      # Taux de mortalité des renards
alpha=0.48; beta=0.86; gamma=1.2400000000000002; delta=1.2400000000000002
step = 0.001     # Intervalle de temps
time = [0]     # Temps initial

# Listes initiales pour les populations simulées
rabbit = [1]  # Population initiale de lapins
fox = [2]     # Population initiale de renards

# Simulation avec boucle
for _ in range(100_000):
    dt = time[-1] + step
    dx = (rabbit[-1] * (alpha - beta * fox[-1])) * step + rabbit[-1]
    dy = (fox[-1] * (delta * rabbit[-1] - gamma)) * step + fox[-1]
    time.append(dt)
    rabbit.append(dx)
    fox.append(dy)

# Conversion des résultats en arrays
time = np.array(time)*1000
rabbit = np.array(rabbit) * 1000  # Échelle pour les lapins
fox = np.array(fox) * 1000       # Échelle pour les renards

# Ajustement des données simulées pour correspondre aux données réelles
simulated_rabbit = rabbit[::30][:1000]  # Échantillonnage tous les 100 points et garder 1000 valeurs
simulated_fox = fox[::30][:1000]
real_lapin = df_cp["lapin"].values[:1000]  # Limitation aux 1000 premières valeurs
real_renard = df_cp["renard"].values[:1000]

# Calcul de l'erreur quadratique moyenne
def MSE(approx, real):
    mse = np.mean((approx - real) ** 2)
    return mse

mse_rabbit = MSE(simulated_rabbit, real_lapin)
mse_fox = MSE(simulated_fox, real_renard)
print(mean_squared_error(df_cp["lapin"],simulated_rabbit))

print(f"Erreur quadratique moyenne pour les lapins : {mse_rabbit:.2f}")
print(f"Erreur quadratique moyenne pour les renards : {mse_fox:.2f}")

# Création d'un fichier CSV avec les données simulées
simulated_data = pd.DataFrame({
    "days": range(1, 1001),
    "simulated_rabbit": simulated_rabbit,
    "simulated_fox": simulated_fox
})
simulated_data.to_csv("simulated_populations.csv", index=False)
print("Fichier CSV des données simulées créé avec succès.")

# Visualisation des populations
plt.figure(figsize=(12, 8))
plt.plot(df_cp["days"], real_lapin,'r--' ,label="Lapins réels")
plt.plot(df_cp["days"], real_renard,'b--' , label="Renards réels")
plt.plot(range(1, 1001), simulated_rabbit, label="Lapins simulés", color="red")
plt.plot(range(1, 1001), simulated_fox, label="Renards simulés", color="blue")
plt.xlabel("Jours")
plt.ylabel("Population")
plt.title("Évolution des populations de lapins et de renards")
plt.legend()
plt.ylim([0, 5000])
plt.grid()
plt.show()
