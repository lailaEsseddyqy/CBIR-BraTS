# verify_multimodal.py
"""
Vérifie que la recherche multimodale retourne
bien des résultats qui respectent les filtres demandés.
"""
import sys, torch
sys.path.append(".")

from src.recherche.multimodal_search import (
    MultimodalSearchEngine, MultimodalQuery, PatientFilter
)
from src.db.connections import get_slices_collection

engine    = MultimodalSearchEngine()
mongo_col = get_slices_collection()

# ─────────────────────────────────────────────────────────
# Test 1 — Hommes + 50 ans + FLAIR
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 : Hommes + âge >= 50 + modalité FLAIR")
print("="*60)

dummy = torch.rand(1, 128, 128)
query = MultimodalQuery(
    image_tensor   = dummy,
    k              = 10,
    patient_filter = PatientFilter(
        sexe     = "Homme",
        age_min  = 50,
        modalite = "flair",
    ),
)
results = engine.search(query)

print(f"\n{'Rang':<5} {'Score':<8} {'Sexe':<8} {'Âge':<6} "
      f"{'Modalité':<10} {'Filtre OK?'}")
print("-" * 60)

errors = 0
for r in results:
    sexe_ok     = r.sexe     == "Homme"
    age_ok      = r.age      >= 50
    modalite_ok = r.modalite == "flair"
    all_ok      = sexe_ok and age_ok and modalite_ok
    if not all_ok:
        errors += 1
    flag = "✅ OK" if all_ok else "❌ ERREUR"
    print(f"{r.rank:<5} {r.score:<8} {r.sexe:<8} {r.age:<6} "
          f"{r.modalite:<10} {flag}")

print(f"\nRésultat : {len(results)-errors}/{len(results)} "
      f"résultats corrects")

# ─────────────────────────────────────────────────────────
# Test 2 — Femmes + CHU Casablanca
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2 : Femmes + CHU Casablanca")
print("="*60)

query2 = MultimodalQuery(
    image_tensor   = dummy,
    k              = 10,
    patient_filter = PatientFilter(
        sexe    = "Femme",
        hopital = "CHU Casablanca",
    ),
)
results2 = engine.search(query2)

print(f"\n{'Rang':<5} {'Sexe':<8} {'Hôpital':<25} {'Filtre OK?'}")
print("-" * 55)

errors2 = 0
for r in results2:
    sexe_ok    = r.sexe    == "Femme"
    hopital_ok = r.hopital == "CHU Casablanca"
    all_ok     = sexe_ok and hopital_ok
    if not all_ok:
        errors2 += 1
    flag = "✅ OK" if all_ok else "❌ ERREUR"
    print(f"{r.rank:<5} {r.sexe:<8} {r.hopital:<25} {flag}")

print(f"\nRésultat : {len(results2)-errors2}/{len(results2)} "
      f"résultats corrects")

# ─────────────────────────────────────────────────────────
# Test 3 — Âge entre 30 et 40 ans
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3 : Âge entre 30 et 40 ans")
print("="*60)

query3 = MultimodalQuery(
    image_tensor   = dummy,
    k              = 10,
    patient_filter = PatientFilter(
        age_min = 30,
        age_max = 40,
    ),
)
results3 = engine.search(query3)

print(f"\n{'Rang':<5} {'Âge':<6} {'Filtre OK?'}")
print("-" * 30)

errors3 = 0
for r in results3:
    age_ok = 30 <= r.age <= 40
    if not age_ok:
        errors3 += 1
    flag = "✅ OK" if age_ok else "❌ ERREUR"
    print(f"{r.rank:<5} {r.age:<6} {flag}")

print(f"\nRésultat : {len(results3)-errors3}/{len(results3)} "
      f"résultats corrects")

# ─────────────────────────────────────────────────────────
# Résumé global
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RÉSUMÉ GLOBAL")
print("="*60)
total_ok    = (len(results)-errors) + (len(results2)-errors2) + (len(results3)-errors3)
total_tests = len(results) + len(results2) + len(results3)
print(f"  Total correct : {total_ok}/{total_tests}")
pct = round(total_ok / max(total_tests, 1) * 100, 1)
print(f"  Précision     : {pct}%")
if pct == 100:
    print("  ✅ Tous les filtres fonctionnent parfaitement !")
elif pct >= 80:
    print("  🟡 Quelques anomalies à vérifier")
else:
    print("  ❌ Problème dans les filtres")