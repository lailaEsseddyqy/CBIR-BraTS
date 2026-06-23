import matplotlib.pyplot as plt
import numpy as np

def plot_precision_at_k():
    # Données issues de votre Tableau 4.2
    k_values = [1, 5, 10]
    baseline_p = [0.68, 0.62, 0.58]
    radimagenet_p = [0.82, 0.78, 0.75]
    combine_p = [0.86, 0.84, 0.81]

    plt.figure(figsize=(8, 5))
    
    # Tracé des courbes avec des marqueurs distincts
    plt.plot(k_values, baseline_p, marker='o', linestyle='--', color='#e74c3c', label='Baseline (256D)', linewidth=2)
    plt.plot(k_values, radimagenet_p, marker='s', linestyle='-.', color='#3498db', label='RadImageNet (2048D)', linewidth=2)
    plt.plot(k_values, combine_p, marker='^', linestyle='-', color='#2ecc71', label='Mode Combiné', linewidth=2.5)

    # Esthétique du graphique
    plt.title("Évolution de la Précision au rang K (P@K)", fontsize=14, fontweight='bold')
    plt.xlabel("Rang K (Profondeur de recherche)", fontsize=12)
    plt.ylabel("Précision", fontsize=12)
    plt.xticks(k_values) # Forcer l'affichage de 1, 5, 10
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11)
    
    # Sauvegarde
    plt.savefig('fig4_1_precision_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 4.1 générée : fig4_1_precision_at_k.png")

# Décommentez pour tester :
plot_precision_at_k()