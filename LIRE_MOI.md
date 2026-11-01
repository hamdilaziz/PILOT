# Notebooks de fine-tuning PILOT

Cette mise à jour s'applique après le premier fichier PILOT_training_update.patch.
L'archive contient les sources pour consultation et le patch qui installe tous les changements.

Place le ZIP à côté du dossier PILOT. Depuis la racine du dépôt PILOT :

```bash
python -m zipfile -e ../PILOT_finetuning_notebooks.zip ../PILOT_notebooks
git apply --check ../PILOT_notebooks/PILOT_notebooks_update.patch
git apply ../PILOT_notebooks/PILOT_notebooks_update.patch
uv pip install -e ".[notebooks]"
uv run jupyter lab
```

Ouvre examples/finetune_fatura.ipynb ou examples/finetune_hiertext_ocr.ipynb.
Sélectionne le kernel de l'environnement dans lequel PILOT et les dépendances ont été installés.
Les poids publics et le tokenizer doivent être présents dans checkpoints/ avant la partie entraînement.

Organisation : préparation réutilisable, audit et statistiques, entraînement/reprise,
évaluation sur test et exemple de prédiction. Les deux notebooks et les explications du dépôt
sont en anglais. Aucun tracking MLflow ou W&B.

FATURA utilise les champs NUMBER, DATE, DUE_DATE, SUB_TOTAL, TAX et TOTAL dans leur texte
annoté d'origine, libellés inclus. Le split stratégie 1 des auteurs évalue de nouvelles instances
sur des gabarits connus; les doublons à fond coloré ne sont pas ajoutés.

HierText illustre l'OCR de lignes lisibles non verticales, recadrées depuis les annotations
et gardées dans les splits officiels des images sources. Il ne mesure pas l'OCR end-to-end
sur les pages entières. Le téléchargement des images s'arrête au nombre demandé.

Validation : 20 tests CPU passent; toutes les cellules Python sont exécutées dans des processus
isolés avec un modèle mBART réduit. La préparation a aussi été vérifiée sur les données réelles
avec 400/80/80 exemples pour chaque notebook. L'entraînement PILOT complet sur GPU reste à tester.
