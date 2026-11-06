# Benchmark cold start v2

Il benchmark aggiornato usa una sola configurazione, un solo valutatore e lo stesso
catalogo candidato per tutti i sistemi. Il risultato principale è **NDCG@10 B a
k=10**; Recall@10 e Hit Rate@10 completano il confronto. Una vittoria di
`bayesian_mean` è un risultato possibile, non un errore da correggere a posteriori.

Questa implementazione è stata verificata con dati sintetici e provider/retrieval
fittizi. I notebook non sono stati eseguiti sui dati MovieLens reali. Nel checkout
usato per lo sviluppo non sono presenti i dataset o il protocollo sorgente.

## Configurazione e responsabilità

- `configs/recommendation_coldstart_v2.json`: versione, sorgenti, seed, griglie,
  modalità pilota, encoder, collection e budget LLM.
- `src/movie_rag/recommendation_benchmark.py`: importazione e controlli del
  protocollo, manifest, cache, metriche A/B, macro per utente e bootstrap.
- `src/movie_rag/recommendation_models.py`: baseline aggregate, MF, fold-in e
  scoring GTE con/senza prior.
- `src/movie_rag/recommendation_experiments.py`: preparazione degli embedding,
  selezione sulla validation, esecuzione e ripresa dei run, confronti e report.

I notebook `recommendation_01`, `_02`, `_03`, `_04`, `_05` e le diagnostiche sono
interfacce sottili a questi moduli; non contengono copie dei modelli o delle
formule delle metriche. Gli output incorporati delle vecchie esecuzioni sono stati
rimossi, per evitare che appaiano come risultati della nuova implementazione.
I file precedenti sotto `datasets/processed/recommendation_protocol/` non vengono
modificati. Le precedenti versioni del codice restano nella cronologia Git.

## Dati richiesti e importazione

Prima di iniziare servono:

1. Il protocollo originale `rs_coldstart_full`, con `protocol_manifest.json`, i
   cinque CSV `inputs/{catalog_movie_ids,train_ratings,scenarios,scenario_levels,
   visible_interactions}.csv` e i quattro CSV privati
   `{user_partitions,private_profile,private_fixed_gt,private_additional}.csv`.
2. Il catalogo `dataset_enriched_movielens_tmdb_full.csv` identico a quello
   registrato nel manifest originale e `datasets/raw/ml-latest-small/movies.csv`.
3. Le dipendenze del progetto. Per embedding e query encoder serve anche
   `sentence-transformers`; per le fasi live servono MCP, SDK del provider e Qdrant.
   Le credenziali vengono lette dall'ambiente, non dai notebook o dagli artefatti.

Il notebook 01 copia i file originali senza cambiarne le assegnazioni casuali e
crea `rs_coldstart_v2`. Verifica gli utenti train/validation/test, lo split 80/10/10
per strato, tre repliche, i dieci elementi del profilo, i venti target legacy,
l'assenza di sovrapposizioni e gli esatti prefissi rivelati. Non seleziona gli
utenti in funzione del numero di positivi nascosti. Non genera una partizione
sostitutiva quando manca quella originale.

Manifest e artefatti sono immutabili: una ripetizione identica è consentita; una
differenza di contenuto viene rifiutata. Cambiando input, codice o configurazione,
usare un nuovo `protocol_id`. Se cambia la rappresentazione degli embedding,
usare anche una nuova collection e un corrispondente profilo del benchmark.

## Ordine di esecuzione

Ogni notebook parte con il proprio flag di esecuzione a `False`. L'importazione
dei moduli non avvia training o richieste esterne. Impostare a `True` soltanto il
flag della fase desiderata.

| Fase | Notebook | Operazione |
| --- | --- | --- |
| 1 | `recommendation_01_protocol_10_20` | `RUN_PREPARE`: importa e valida la sorgente |
| 2 | `recommendation_01b_v2_embeddings` | `RUN_EMBEDDINGS`: codifica il catalogo pubblico e popola la nuova collection |
| 3 | `recommendation_02_baselines_mf_gte` | `RUN_BASELINES`: baseline nella modalità scelta |
| 4 | `recommendation_04_llm_semantic_profile` | `RUN_LLM`: profili, GTE e GTE+prior nella stessa modalità |
| 5 | `recommendation_05_pipeline_lineare_benchmark` | `RUN_LIVE`: pipeline lineare nella stessa modalità |
| 6 | `recommendation_03_diagnostica_ranking` | `READ_REPORTS`: legge i confronti compatibili |

Le modalità sono `pilot_validation`, `full_validation`, `test`.

- Il pilot usa al massimo sei utenti validation, replica zero. Serve a verificare
  il percorso, non congela i parametri e non abilita il test.
- `full_validation` usa tutte le repliche degli utenti validation. Congela le
  selezioni nei file `selections/{baselines,llm_semantic,pipeline}.json`.
- `test` richiede la selezione full-validation compatibile del relativo sistema.
  Non seleziona o modifica iperparametri.

Per ogni modalità, eseguire prima le baseline: le fasi LLM le richiedono per
confrontarsi esattamente sugli stessi utenti e livelli. Prima di spendere il
budget vengono controllati gli artefatti necessari. `MAX_NEW_CALLS` limita le
nuove chiamate ai profili semantici oppure le nuove **invocazioni complete del
workflow** lineare, che possono contenere sia Planner sia Answer. Per la
full-validation il limite predefinito di 40 può risultare insufficiente: il run
si ferma prima delle nuove chiamate, indicando il budget da aumentare. Le cache
sono distinte per modalità e sottoinsieme. Aumentare `MAX_NEW_CALLS` nella cella
del notebook, senza cambiare la configurazione già congelata nel manifest.

## Rappresentazione e retrieval

I documenti sono costruiti da una allowlist di titolo, generi, overview, tagline,
keyword, cast, crew, collection e società di produzione. Non usano `ml_tags`,
testi aggregati MovieLens, rating o conteggi di interazioni, anche se tali colonne
sono presenti nel CSV. Il titolo MovieLens è ammesso solo come fallback pubblico
quando manca quello TMDb.

Gli embedding normalizzati vengono salvati sotto `artifacts/embeddings/` del nuovo
protocollo. La collection dedicata è
`movies_rs_coldstart_v2_gte_no_interactions_v1`. L'upsert può riprendere una
collection parziale solo se i punti esistenti hanno lo stesso fingerprint; non
elimina collection. La pipeline rifiuta collection incomplete o incompatibili.

Il profilo `movielens_coldstart_benchmark_v2` usa ricerca esatta e trenta candidati
senza reranker. Il dominio dedicato ammette soltanto `filters={}`. Dopo il
retrieval si escludono gli ID visibili e si scelgono i primi dieci restanti.
Il profilo FAST generale dell'applicazione conserva i suoi limiti originari.

## Sistemi, selezione e correzioni

- `bayesian_mean`: media con shrinkage, prior 15, calcolata sui soli rating train.
- `most_popular`: conteggio dei rating train; parità risolta per movieId crescente.
- `mf_explicit_foldin`: griglia rank `{8,24}`, epoche `{18,60}`, regolarizzazione
  `{0.02,0.1}`, fold-in `{0.1,1,5,20}`, learning rate `0.008`.
- `mf_bias_only`: modello rank zero, con le stesse griglie applicabili di epoche,
  regolarizzazione e fold-in. Isola il contributo dei fattori latenti.
- `gte_direct`, `gte_prior`, `llm_gte_direct`, `llm_gte_prior`: penalità mista
  `{0,0.2,0.5,0.8}`, penalità solo-negativi `{0,0.35,0.7}`. Le varianti con prior
  esplorano peso semantico `{0,0.25,0.5,0.75,1}` nel ramo con positivi.
- `llm_pipeline_linear_retrieval`: ranking restituito da MCP. La risposta Answer
  è salvata nella traccia e non viene trasformata implicitamente nel ranking.

Con positivi, lo score ibrido è `(1-w)*z(bayesian_mean) + w*z(semantic)`.
Il caso `w=1` mantiene il ranking della baseline semantica pura. Con soli negativi
si usa `z(bayesian_mean) - 3*beta*cosine_negative`; senza feedback significativo
si usa il prior. `beta` è selezionato separatamente da `alpha` del ramo misto.

Gli item senza rating train hanno fattori e bias MF zero. Se uno di essi compare
nel profilo può contribuire alla stima del bias utente, ma non introduce fattori
casuali. A `k=0` i metodi personalizzati condividono il fallback bayesiano;
`most_popular` rimane una baseline distinta.

La selezione usa la media di NDCG B sui livelli `{1,3,5,10}`, dopo la media delle
repliche per utente. Le parità sono risolte per identificativo di configurazione
ordinato. Il pilot non viene usato come sostituto della full-validation.

## Errori, cache e report

Ogni run include fingerprint di protocollo, input, codice, configurazione,
provider, prompt/rappresentazione e sottoinsieme. I profili LLM sono validati anche
quando provengono dalla cache; non si accede a `trace` come attributo di un dict.
Le temperature esplicite raggiungono entrambi gli agenti; `None` lascia il default
del provider.

Le chiamate riuscite sono salvate immediatamente in JSONL. Gli errori registrano
il tipo di eccezione e interrompono il run; una nuova invocazione identica riprende
dal primo elemento non completato. Un errore non viene riutilizzato come risposta
né trasformato in un risultato vuoto. `no_results` e `needs_clarification` sono
invece stati validi e restano nel denominatore del confronto con guadagno zero
quando esistono target positivi. Non vengono aggiunti suggerimenti di Popularity
per riempire artificialmente liste corte.

I risultati sono sotto
`evaluations/<esperimento>/<modalità>/<fingerprint>/` e comprendono:

- ranking e stato di tutti i run attesi;
- metriche A/B per scenario e per utente, macro-medie e numero di utenti con
  NDCG definita;
- `primary_B_k10.csv`, differenze appaiate rispetto a `bayesian_mean` e intervalli
  bootstrap al 95% da 2.000 ricampionamenti di utenti;
- copertura dei giudizi nella top 10, conteggi separati di positivi/negativi/neutri/
  non giudicati, lunghezza e carenza delle liste;
- copertura del catalogo e recall nelle fasce train `0`, `1–4`, `5–19`, `20–99`,
  `100+`, con numero di target per fascia;
- manifest con modalità, sottoinsieme, limiti metodologici e hash dei CSV.

Senza positivi la metrica è indefinita (`NaN`), non zero. Con un solo utente
valutabile l'intervallo bootstrap rimane indefinito. Non contare nuovamente le
baseline ripetute nei confronti salvati da esperimenti diversi.

## Verifica locale

I test sintetici sono in `tests/test_recommendation_benchmark_v2.py`, oltre ai
test di compatibilità dell'helper. In un ambiente con le dipendenze installate:

```sh
python -m pytest tests/test_recommendation_benchmark_v2.py tests/test_recommendation_05_helpers.py tests/test_resource_integration.py tests/test_resource_llm_integration.py
```

Le prove coprono anche importazione immutabile, modifiche ai fingerprint, riuso e
rifiuto delle cache, validazione full-validation → test, profili semantici fittizi,
metriche calcolabili a mano, retrieval vuoto e ripresa dopo errore. Non richiedono
dataset MovieLens, servizi attivi, credenziali o download di modelli.

## Interpretazione scientifica

Lo schema B recupera i positivi noti esclusi dal limite legacy dei venti target.
Non elimina la selezione insita nei rating osservati: un film senza giudizio ha
guadagno osservato zero, non una preferenza negativa accertata. Il tasso di
positivi fra i giudicati e le fasce di frequenza sono diagnostiche, non stimatori
automaticamente imparziali della qualità su tutto il catalogo.

I target B cambiano con k: il confronto principale è fra modelli allo stesso
livello; la curva non isola l'effetto della personalizzazione. Le repliche dello
stesso utente sono correlate e vengono mediate prima del bootstrap. Il test già
ispezionato resta esplorativo anche cambiando codice o seed: una verifica
confermativa richiede dati non usati per scegliere il metodo.
