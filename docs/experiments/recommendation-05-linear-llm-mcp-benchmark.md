> **Versione precedente.** I notebook correnti implementano il [benchmark cold start v2](recommendation-coldstart-v2.md). Questo documento conserva configurazioni e percorsi storici.

# Recommendation 05: benchmark della pipeline LLM/MCP lineare

> **Provenienza:** `README_recommendation_05_pipeline_lineare_benchmark.md` · **Tipo:** documentazione tecnica del progetto.
> **Collocazione nel percorso:** Benchmark del ranking restituito dal retrieval MCP; Answer analizzato separatamente.
> **Nota:** i contenuti tecnici sottostanti sono stati conservati dalla documentazione fornita;
> non costituiscono una nuova verifica dell’esecuzione del codice nel repository corrente.

## Contenuti

- `notebooks/recommendation_05_pipeline_lineare_benchmark.ipynb` — notebook di sperimentazione.
- `notebooks/recommendation_05_pipeline_helpers.py` — template deterministici, estrazione dei ranking strutturati da MCP, valutazione A/B.
- `tests/test_recommendation_05_helpers.py` — test unitari offline.

Non sovrascrive file esistenti in `src/movie_rag`, né Qdrant, né gli artifact dei notebook precedenti. Importa le factory originali del workflow lineare documentate nei notebook 11 e `orchestration_iterative_v1`.

## Installazione

1. Copiare **entrambi** i file di `notebooks/` nella cartella `<repo>/notebooks/` del progetto. Senza il `.py` di supporto il notebook non può partire.
2. Assicurarsi che siano presenti le configurazioni `configs/llm_profiles_v1.json`, `configs/resource_domains_v1.json`, `configs/resource_mcp_profiles_v1.json`, il package `src/movie_rag`, Qdrant attivo e il protocollo `rs_coldstart_v1` prodotto dal notebook 01.
3. Iniziare da `RUN_LIVE = False`. Le prime celle generano le richieste deterministiche e verificano configurazioni senza chiamate API.
4. Quando la preview è soddisfacente, impostare `RUN_LIVE = True` con `RUN_MODE='pilot_validation'`. Default: 6 utenti validation, una replica ciascuno, quattro livelli non nulli = **24 run di workflow** (ogni run può chiamare sia Planner sia Answer LLM, oltre al MCP). `MAX_NEW_CALLS` limita le esecuzioni logiche, non il totale delle chiamate HTTP o retry interni.
5. I provider attuali della factory sono configurati tramite `gemini_free` (o altro profilo disponibile nel tuo registry). Viene caricata la `.env` locale se `python-dotenv` è installato; non pubblicare chiavi nei risultati.
6. Per usare tutti gli utenti di validation: `RUN_MODE='full_validation'` e alzare consapevolmente `MAX_NEW_CALLS`. Il test resta protetto da `RUN_MODE='test'` **e** `ALLOW_TEST=True` dopo aver congelato il metodo.

### Adattare il template

La funzione di default `title_rating_v1` genera un testo italiano usando soltanto titoli, `movieId` e rating numerici rivelati. La logica vive nel **registro `TEMPLATES`** del file helper e non è affidata a un LLM aggiuntivo.

Per prove future, registrare un nuovo template con un **nome versione diverso**, ad esempio `title_rating_genres_v2`, senza modificare retroattivamente `title_rating_v1`: il fingerprint della run include anche l'hash del modulo helper e delle configurazioni/sorgenti del workflow. I vecchi risultati rimarranno riproducibili.

### Cosa misura esattamente

La pipeline chiamata dal notebook è:

**Richiesta deterministica → Planner LLM → MCP `search_resources` → RetrievalService GTE/Qdrant → Answer LLM**.

L'Answer LLM può produrre meno di dieci raccomandazioni. Le metriche principali **non** valutano il suo testo: sono calcolate sui primi dieci ID validi presenti in `state.search_response.results`, rispettando l'ordine del retriever e scartando gli item già rivelati. Per questo si richiedono **30 candidati** al MCP; dopo il filtro dei film già visti ne vengono valutati al massimo dieci. Quando ne risultano meno, il caso è registrato (`shortfall`), e le posizioni mancanti contribuiscono con gain zero, senza fallback implicito.

Si salva separatamente l'elenco di ID presente in `state.final_answer.recommendations` e il testo finale, per poter studiare in una fase successiva quanto la selezione dell'Answer differisca dal retrieval. Non attribuire le metriche del retrieval alla qualità dell'Answer LLM.

A `k=0` non c'è alcun profilo: si usa lo **stesso ranking Popularity** del benchmark precedente, senza invocare LLM. Il dialogo tra due LLM per il vero cold-start conversazionale non è incluso.

### Metriche e set A/B

Il notebook conserva `scenario_id`, `k`, `movieId`, `rank`, `score` e produce `ndcg@10`, `recall@10`, `hr@10` per A (20 rating fissi) e B (tutti i rating nascosti). I confronti vengono effettuati sugli stessi scenari con `popularity`, `gte_direct` e `mf_explicit_foldin`, senza riallenarli.

**Separazione delle informazioni:** le celle di generazione delle richieste e invocazione della pipeline leggono soltanto `inputs/` e `movies.csv`. `private_evaluation/` viene letto esclusivamente nella cella di valutazione dopo che i ranking sono già stati salvati.

### Output

Sotto `datasets/processed/recommendation_protocol/rs_coldstart_v1/evaluations/llm_pipeline_linear_v1/<RUN_MODE>/`:

- `workflow_runs.jsonl`: checkpoint delle esecuzioni complete con query, query inviata al MCP, risposta, ranking, stato, tempi e hash configurazione.
- `rankings_llm_pipeline_mcp.csv`: top-K strutturato, formato comparabile con le baseline.
- `answer_recommendation_ids.csv`: ID suggeriti nella risposta testuale (non usati nelle metriche principali).
- `pipeline_traces.csv`: input e output necessari per la diagnostica qualitativa.
- `metrics_per_scenario_A_B.csv`, `metrics_per_user_A_B.csv`, `macro_A_B.csv`.
- `manifest_run.json`: informazioni di riproducibilità e numero di ranking incompleti.

## Test svolti

- Validazione/compilazione di tutte le celle del notebook.
- Test unitari del template e della trasformazione degli output MCP in top-K, incluse risposte dell'Answer con meno elementi, filtri dei film già noti, deduplicazione, ground truth A/B, liste corte e liste vuote.
- **Test end-to-end simulato** con dataset, provider LLM, server MCP e workflow fittizi: esecuzione di tutte le celle, export dei ranking e calcolo di A/B con quattro sistemi. Superato.
- **Non** sono stati eseguiti Qdrant né API LLM reali. L'interfaccia è stata implementata secondo i notebook 11 e `orchestration_iterative_v1` disponibili. È necessaria una verifica di compatibilità iniziale sulla tua `src/` attuale, soprattutto se modificata dopo ottobre.

## Cautela scientifica

Il retrieval può produrre film senza rating noto: non sono necessariamente negativi. Inoltre, i documenti già indicizzati in Qdrant possono contenere user tags MovieLens aggregati precedenti allo split e quindi un possibile leakage. Il confronto rimane esplorativo finché questa limitazione non viene isolata. Il vecchio test è già stato ispezionato: nuove ottimizzazioni vanno fatte sulla validation, poi confermate su un test non consultato.
