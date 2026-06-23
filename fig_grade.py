import matplotlib.pyplot as plt
import numpy as np

def plot_stratified_precision():
    # VOS VRAIES DONNÉES STRATIFIÉES (P@5)
    grades = ['Grade II', 'Grade III', 'Grade IV']
    
    # Ordre : [Grade II, Grade III, Grade IV] extrait de vos logs
    p_at_5_baseline = [0.0667, 0.0533, 0.8667]
    p_at_5_radimage = [0.0533, 0.0533, 0.8667]
    p_at_5_combine  = [0.0667, 0.0533, 0.8800]

    x = np.arange(len(grades))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(9, 6))
    
    rects1 = ax.bar(x - width, p_at_5_baseline, width, label='Baseline', color='#e74c3c')
    rects2 = ax.bar(x, p_at_5_radimage, width, label='RadImageNet', color='#3498db')
    rects3 = ax.bar(x + width, p_at_5_combine, width, label='Mode Combiné', color='#2ecc71')

    ax.set_ylabel('Précision (P@5)', fontsize=12)
    ax.set_title('Précision P@5 stratifiée par Grade Tumoral', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grades, fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # L'axe Y va de 0 à 1 (100%)
    ax.set_ylim(0, 1.0)

    plt.savefig('fig4_3_precision_stratified.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 4.3 générée avec succès.")

# Exécuter la fonction
plot_stratified_precision()