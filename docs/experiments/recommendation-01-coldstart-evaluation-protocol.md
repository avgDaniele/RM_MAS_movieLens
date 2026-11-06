> **Versione precedente.** I notebook correnti implementano il [benchmark cold start v2](recommendation-coldstart-v2.md). Questo documento conserva configurazioni e percorsi storici.

# Recommendation 01: protocollo sperimentale cold-start A/B

> **Provenienza:** `README_recommendation_protocol(1).md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Split, scenari, rating visibili, ground truth nascosta e formato dei ranking.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## File
`recommendation_01_protocol_10_20.ipynb` — notebook autonomo; non modifica né importa `src/movie_rag`, MCP, LLM o orchestratori.

## Dove collocarlo
Copia il notebook dentro `<project_root>/notebooks/`. Le sole dipendenze sono `pandas`, `numpy`, `IPython`, oltre al kernel Python. Non sono necessarie GPU o API.

Richiede:
- `<project_root>/datasets/raw/ml-latest-small/ratings.csv`
- `<project_root>/datasets/raw/ml-latest-small/movies.csv`
- `<project_root>/datasets/processed/dataset_enriched_movielens_tmdb.csv`

Se i percorsi differiscono, imposta `PROJECT_ROOT_OVERRIDE` nella prima cella di configurazione.

## Output
Il notebook crea solo file sotto:
`datasets/processed/recommendation_protocol/rs_coldstart_v1/`

- `inputs/`: `catalog_movie_ids.csv`, `train_ratings.csv`, `scenarios.csv`, `scenario_levels.csv`, `visible_interactions.csv`.
- `private_evaluation/`: `private_profile.csv` (10 rating per simulatore), `private_fixed_gt.csv` (20 target A), `private_additional.csv` (rating rimanenti), oltre ai metadati di verifica. Non importare questi file nei modelli.
- `protocol_manifest.json`: seed, dimensioni, conteggi e hash delle sorgenti.

I modelli devono ricevere soltanto `inputs/train_ratings.csv` durante il training globale e le righe pertinenti di `inputs/visible_interactions.csv` al livello `k`; il dataset degli item è il catalogo.

## Come valutare i ranking
Quando saranno disponibili, salva `predictions/ranking_results.csv` con:
`scenario_id,k,model,movieId,rank,score` (`score` facoltativo).

Riesegui l'ultima cella di valutazione del notebook. Gli stessi ranking vengono confrontati con:
- A: solo i 20 target fissi;
- B: tutti i rating non ancora rivelati (20 target, profilo privato non rivelato, e rating extra).

Il valutatore rifiuta ranking con item già rivelati, item duplicati o item fuori catalogo. I risultati vengono salvati in `evaluations/` con macro-medie per utente.

## Note metodologiche
- Gli utenti validation/test non sono presenti in `inputs/train_ratings.csv`.
- Film senza rating = preferenza ignota, non giudizio negativo.
- A/B hanno numeri di target rilevanti diversi: non confrontare direttamente i valori assoluti di Recall@10 fra A e B.
- I risultati dell'auto-test sono **fittizi**, non costituiscono valutazioni MF/GTE.
- Gli embedding contenenti tag o statistiche aggregate dai rating MovieLens degli utenti di test possono produrre leakage: per GTE conviene costruire rappresentazioni indipendenti dalle interazioni degli utenti di test.
- Gli split casuali sono diagnostici e replicati; i test cronologici o la cross-validation multi-fold possono essere aggiunti dopo, mantenendo il protocollo già definito.

## Stato del codice LLM
Il notebook è indipendente dai componenti `src/movie_rag/resource_*`, dalle configurazioni dei profili MCP e da W0/W1/W2. Non è una modifica da applicare ai notebook dell'orchestrazione.
