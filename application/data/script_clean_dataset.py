import pandas as pd

input_file = "train.csv"       # ton fichier brut
output_file = "cleaned_dataset.csv"  # fichier propre

# Lire CSV en ignorant les lignes problématiques
df = pd.read_csv(
    input_file,
    on_bad_lines="skip",      # ignore les lignes corrompues
    engine="python"           # parser plus flexible
)

columns_to_keep = ["textID", "text", "selected_text", "sentiment"]

missing_cols = [c for c in columns_to_keep if c not in df.columns]
if missing_cols:
    print("Colonnes manquantes :", missing_cols)
else:
    df_clean = df[columns_to_keep]

    df_clean.to_csv(output_file, index=False)
    print(f"Fichier nettoyé → {output_file}")
