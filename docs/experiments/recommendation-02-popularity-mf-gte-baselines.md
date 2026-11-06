> **Versione precedente.** I notebook correnti implementano il [benchmark cold start v2](recommendation-coldstart-v2.md). Questo documento conserva configurazioni e percorsi storici.

# Recommendation 02: benchmark Popularity, MF e GTE diretto

> **Provenienza:** `README_recommendation_02.md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Baseline statistiche/collaborative/semantiche sul protocollo rs_coldstart_v1.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## Cosa contiene

- `recommendation_02_baselines_mf_gte.ipynb`: notebook autonomo, **nessuna modifica a `src/`** e nessuna nuova inferenza GTE.
- Input già generati da `recommendation_01_protocol_10_20.ipynb` nella cartella `datasets/processed/recommendation_protocol/rs_coldstart_v1`.
- Confronto sullo stesso catalogo e sugli stessi scenari con `k = 0, 1, 3, 5, 10` rating visibili.
- MF su rating espliciti, stimando il nuovo profilo con `fold-in` dei soli rating visibili; selection di `fold_reg` sulla validation.
- GTE mediante vettori precalcolati e aggregazione semantica dei film apprezzati, con penalità sui non apprezzati; selection di `negative_alpha` sulla validation.
- Ranking sul catalogo completo, esclusi i soli film già rivelati. Lo stesso ranking serve per metriche A (20 target fissi) e B (tutti i rating nascosti, non solo i 20).

## Installazione / Esecuzione

1. Copiare il notebook in `<root_progetto>/notebooks/`.
2. Verificare la presenza dei CSV del protocollo già prodotti da `recommendation_01` sotto `datasets/processed/recommendation_protocol/rs_coldstart_v1/`.
3. Aprire il notebook; nella prima cella di configurazione scegliere `GTE_SOURCE='qdrant'` per usare i vettori GTE già presenti in Qdrant o `GTE_SOURCE='npz'` per leggere un file esistente contenente `movie_ids` e `vectors`.
4. Se si usa Qdrant, controllare `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_MOVIE_ID_KEY` e, se necessario, `QDRANT_VECTOR_NAME`.
   - Se `QDRANT_COLLECTION=None`, la ricerca automatica funziona solo quando c'è **esattamente una** collezione il cui nome contiene `gte`.
   - Verificare che sia il **modello GTE corretto**, non una vecchia collezione BGE-M3 o da soli 50 film.
   - È richiesta copertura totale degli item del protocollo, senza riaddestrare encoder/embedding.
5. Eseguire le celle in sequenza. Servono `numpy`, `pandas`, `matplotlib`, `IPython`, più `qdrant-client` soltanto nella modalità `qdrant`. Non sono richiesti LLM/GPU.

Se Qdrant non è raggiungibile, la parte Popularity/MF può comunque funzionare e il notebook indica chiaramente che il confronto con GTE è **incompleto**. Dopo aver corretto la configurazione dei vettori, rieseguire dalla sezione 4 in poi.

## File esportati

Il notebook **non modifica** file dell'orchestratore né gli input del protocollo. Salva:

- `predictions/rankings_baselines_validation.csv` — ranking per la selezione dei parametri.
- `predictions/rankings_baselines_test.csv` — ranking dei modelli finali sul test.
- `predictions/ranking_results.csv` — creato soltanto se non esiste già, compatibile con il valutatore del notebook 01.
- `evaluations/baseline_mf_gte/validation_macro_A_B.csv`
- `evaluations/baseline_mf_gte/validation_hyperparameter_selection.csv`
- `evaluations/baseline_mf_gte/test_metrics_per_scenario_A_B.csv`
- `evaluations/baseline_mf_gte/test_metrics_per_user_A_B.csv`
- `evaluations/baseline_mf_gte/test_macro_A_B.csv`
- `evaluations/baseline_mf_gte/model_config.json`

Non adattare i parametri al test: la scelta di `fold_reg` e `negative_alpha` usa soltanto la **validation A**. Per un'aggiunta LLM futura usare lo stesso formato `scenario_id,k,model,movieId,rank,score`.

## Nota metodologica indispensabile

I vettori GTE già calcolati potrebbero incorporare `ml_tags` e statistiche di rating MovieLens ottenuti da tutti gli utenti. **Non basta non includerli nelle query**: fanno parte della rappresentazione del catalogo. Di conseguenza, anche con split utente corretti, il confronto MF/GTE sui vecchi vettori è un esperimento **esplorativo con possibile leakage delle informazioni sui test users**. Un risultato formale leakage-free richiederà eventualmente una nuova indicizzazione con soli metadata esterni al test (es. descrizioni TMDb).

Il protocollo MovieLens resta inoltre soggetto al problema degli item mai valutati: assenza di rating ≠ preferenza negativa. Le metriche A e B non sono direttamente comparabili nei valori assoluti quando il numero di target positivi cambia.

## Test di esecuzione

Verificato il funzionamento end-to-end su un **dataset sintetico**, inclusi protocollo 01, MF, caricamento NPZ GTE, fold-in, selezione iperparametri su validation e doppio calcolo A/B sul test. Le metriche reali dipendono dai CSV e dagli embedding nel tuo progetto.
