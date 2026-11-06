> **Versione precedente.** I notebook correnti implementano il [benchmark cold start v2](recommendation-coldstart-v2.md). Questo documento conserva configurazioni e percorsi storici.

# Recommendation 04: profilo semantico LLM → GTE (fase 2A)

> **Provenienza:** `README_recommendation_04_fase_2A(1).md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Confronto LLM+GTE vs GTE diretto con rating controllati, cache e metriche appaiate.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## Obiettivo scientifico
Confrontare **GTE diretto** (centroidi dei vettori di film già valutati) con **LLM → profilo semantico → GTE**, a parità dei film e rating rivelati, usando il protocollo `rs_coldstart_v1` (k=0,1,3,5,10), lo stesso catalogo dei notebook 01/02 e lo stesso scoring (α negativo già selezionato sulla validation dal notebook 02). **Nessuna conversazione tra due LLM in questa fase**, nessun reranking, nessun Critic/Answer e nessuna ricerca di nuovi iperparametri.

Per ogni output top-10 si valutano entrambi i protocolli: **A** (20 rating nascosti fissi), **B** (tutti i rating non rivelati). Lo stesso ranking viene riusato; nessuna chiamata extra per B.

## Preparazione

1. Copia `recommendation_04_llm_semantic_profile.ipynb` in `<progetto>/notebooks/`.
2. Verifica di avere già eseguito i notebook 01 e 02, con i file presenti sotto `datasets/processed/recommendation_protocol/rs_coldstart_v1/` e la collection Qdrant `movies_d_full_enriched_v2_gte_multilingual_base_v1`.
3. I payload Qdrant devono avere `resource_id` corrispondente a `movieId` MovieLens; è l'impostazione predefinita. Non usare `tmdb_id`.
4. Sono necessari `numpy`, `pandas`, `qdrant-client`, `pydantic>=2`, `sentence-transformers`, `ipykernel` e, per LLM, le dipendenze **già previste nel tuo progetto** (`google-genai` per il provider Gemini oppure `openai`), più facoltativamente `python-dotenv` per leggere `.env`.
5. Assicurati che il package **attuale** `src/movie_rag` esponga `llm_config.load_llm_config`, `role_profile_overrides_from_environment`, `llm_providers.LLMProviderRegistry` e `provider.generate_structured(...)`. Il notebook usa i contratti disponibili nella versione del progetto archiviata a settembre; il sorgente finale di ottobre non era qui accessibile, quindi l'API concreta va verificata nel repository locale. Non copiare una vecchia `src/` dal pacchetto.
6. Le chiavi vengono lette dall'ambiente o dal `.env` del progetto, senza essere salvate nei report. `GTE_MODEL_ID` è **Alibaba-NLP/gte-multilingual-base**, usato SOLO per le query generate, mai per ricalcolare l'indice film.

## Esecuzione graduale e costi

- **Default**: `RUN_MODE='pilot_validation'`, massimo **10 utenti distinti × 1 replica × 4 livelli con informazioni (1,3,5,10)**. Limite predefinito `MAX_NEW_LLM_CALLS=40` (conta le *richieste logiche*, eventuali retry possono aumentare le chiamate API). A `k=0` e con soli rating intermedi si usa il fallback senza LLM.
- **Validation completa**: `RUN_MODE='full_validation'` e aumento esplicito di `MAX_NEW_LLM_CALLS`; non cambiare più prompt/score durante il confronto finale. La cache `semantic_profiles_cache.jsonl` è condivisa fra pilota/validation/test e associa ogni record a scenario, livello, hash dei dati visibili, hash del prompt e profilo del modello; le risposte già ottenute si riutilizzano.
- **Test**: `RUN_MODE='test'` **e** `ALLOW_TEST_RUN=True` soltanto dopo aver congelato la configurazione. Il vecchio test è già stato esaminato e qualsiasi nuova sperimentazione sullo stesso split va qualificata *esplorativa*; per conclusioni confermative future occorre un nuovo test disgiunto o cross-validation con disegno predeterminato.
- I costi monetari non sono stimati dal notebook perché dipendono dal listino del provider. Esporta token, numero chiamate e durata osservata.

## Contratto del profilo semantico

LLM legge per ciascun `scenario_id,k` **solo** i `k` rating di `visible_interactions.csv`, uniti a `movies.csv` per `title` e `genres`. Restituisce un oggetto strutturato:

```json
{"positive_query":"English description of liked content or null", "negative_query":"English description of disliked content or null", "brief_rationale":"Brief grounded explanation"}
```

Il modello non riceve `private_evaluation/`, non vede i giudizi nascosti né i rating degli altri utenti. Validiamo presenza/assenza delle query secondo il numero di rating >=4 e <=2. Le query sono codificate con il modello GTE, poi confrontate con **tutti i film del catalogo** mediante coseno; esclusi soltanto quelli rivelati. Il query encoder carica il modello per codificare le query, ma **non ricalcola il catalogo**.

Formula: con feedback positivi `score=cos(query_pos, movie)-α·cos(query_neg, movie)` se negativi presenti; con soli negativi il fallback è `pop_z-α_neg_only·3·cos(query_neg,movie)`; con soli neutrali o a zero rating si usa la stessa Popularity del notebook 02. L'importanza relativa dei rating all'interno di ciascuna categoria può differire fra le query dell'LLM e i centroidi diretti: è parte del trattamento semantico che si vuole studiare.

## File prodotti

**Non** modifica i file precedenti, l'indice Qdrant, né il codice degli agenti. Scrive sotto:

```
datasets/processed/recommendation_protocol/rs_coldstart_v1/evaluations/llm_gte_phase2a/
  semantic_profiles_cache.jsonl
  pilot_validation/   (oppure full_validation/, test/)
    rankings_llm_gte.csv
    metrics_per_scenario_A_B.csv
    metrics_per_user_A_B.csv
    macro_A_B.csv
    paired_per_scenario.csv
    paired_delta_ndcg10.csv
    llm_call_usage.csv
    run_manifest.json
```

In `paired_delta_ndcg10.csv` il confronto LLM+GTE vs GTE diretto è appaiato per utente/scenario/livello. L'intervallo bootstrap ricampiona **utenti**, non singole repliche. Nel pilota la dimensione campionaria è molto bassa, quindi il CI è puramente orientativo. Le metriche A e B **non** sono confrontabili nei valori assoluti perché cambiano i target rilevanti.

## Test effettuati qui

- Notebook verificato con `nbformat` e compilazione delle celle, inclusi gli `await` asincroni.
- **Smoke test end-to-end con mock** di MovieLens, Qdrant, provider LLM e GTE query encoder: lettura degli input, cache, structured output, retrieval sul catalogo intero, esclusione dei film visibili, fallback `k=0`, calcolo A/B, delta appaiati, export. Tutte le celle sono state eseguite correttamente.
- **Non** è stata effettuata alcuna chiamata ai tuoi servizi reali: il risultato scientifico va ottenuto eseguendo il notebook nel tuo progetto.

## Limiti metodologici

- Gli **embedding dei film già indicizzati contengono potenzialmente user tags MovieLens**, e quindi possono essere contaminati dalle preferenze degli utenti di test; non basta omettere i tag dal prompt LLM. La fase 2A conserva esplicitamente questo limite per confrontabilità con la baseline 02. Un confronto completamente privo di leakage richiederebbe eventualmente un indice ottenuto solo da metadati esterni ai rating di test.
- Il testo che vede l'LLM (`title`, `genres`) non è identico al testo documentale usato nell'indice GTE (che contiene descrizioni più ricche); si mantengono identici **rating, item candidati e punteggio**, ma non la rappresentazione linguistica. Non attribuire in modo causale tutti i cambiamenti alla sola capacità di ragionamento LLM.
- I film senza rating rimangono **unjudged**, non dislike; una baseline popolare può avere vantaggi di esposizione nella valutazione offline.
- La self-similarity tra testo Qdrant e vettore indicizzato è un **controllo diagnostico** dell'encoder, non garanzia completa di uguale preprocessing/instruction/pooling. Se il controllo produce warning, verificare le condizioni originali di embedding prima di interpretare le metriche.
