# Movie RAG · Cold-Start Recommendation · Orchestrazione LLM

Documentazione di orientamento del progetto di tesi: un prototipo di **recommendation / retrieval-grounded LLM** sviluppato con **MovieLens + metadati TMDb** come banco di prova, con l’obiettivo di studiare componenti riutilizzabili in un futuro contesto didattico **WhoTeach**.

> **Come leggere questo README (5 novembre 2026).** È una ricostruzione basata sui dieci README forniti, **non** una verifica del contenuto e della versione effettivamente installata del repository. Descrizioni di test, risultati e stato dell’implementazione restano quelli dichiarati nei documenti di origine. Il codice e gli output di una run nel tuo ambiente sono la fonte di verità per lo stato operativo.

## Da dove partire

Il benchmark di recommendation corrente è descritto nella [guida cold start v2](docs/experiments/recommendation-coldstart-v2.md). I notebook di recommendation ora delegano ai moduli condivisi; i documenti storici sotto mantengono il contesto delle versioni precedenti.

| Cerco… | Documento |
| --- | --- |
| Una mappa di tutti i README, compresi i nomi originali | [Indice dei documenti](docs/INDEX.md) |
| Il percorso di sviluppo e le decisioni tecniche | [Evoluzione del progetto](docs/PROJECT_EVOLUTION.md) |
| **La baseline generica più recente documentata** (`resource_*`) | [Architettura LLM + MCP + risorse](docs/architecture/resource-llm-mcp-workflow.md) |
| La transizione da GTE/Qdrant al MCP generico | [Integrazione GTE / resource MCP](docs/architecture/gte-qdrant-resource-mcp-integration.md) |
| Il prototipo di orchestrazione con cicli e decisioni | [Orchestrazione iterativa v1](docs/orchestration/iterative-orchestration-v1-installation.md) |
| Il protocollo cold-start e i benchmark RS | [Esperimenti di recommendation](docs/INDEX.md#3-esperimenti-di-raccomandazione) |
| Avvertenze su leakage, metriche e confronto tra modelli | [Limiti e criteri sperimentali](docs/RESEARCH_LIMITATIONS.md) |
| La precedente architettura `search_movies` / BGE-M3 | [README storico MovieLens](docs/history/movielens-bge-m3-legacy-architecture.md) |

## Obiettivo e confini

Il lavoro esplora il **cold-start** e la raccomandazione fondata su evidenze recuperate: dai segnali presenti nella richiesta iniziale (o dai pochi rating esplicitamente rivelati) si cerca di recuperare e motivare risorse senza inventarne. L’orientamento della tesi è la **Cold-Case Orchestration**: non soltanto trovare risultati, ma decidere come organizzare il processo di retrieval, verifica e risposta.

Occorre però distinguere tre livelli:

- **Implementazione storica MovieLens:** modelli BGE-M3, Qdrant, MCP `search_movies`, workflow LLM iniziale.
- **Baseline resource-oriented successiva (notebook 10–11):** encoder GTE multilingual, `ResourceDocument`, `RetrievalService`, tool MCP `search_resources` / `get_resources`, agenti `ResourcePlannerAgent` e `ResourceAnswerAgent`, workflow prevalentemente fisso.
- **Evoluzione iterativa v1:** controller con Elicitation → [Retrieval → Validation → Decision] → Answer, con pause/chiarimenti, retry limitati e budget. La documentazione dichiara test offline/mock, **non** validazione end-to-end live.

**WhoTeach non è già integrato:** i README descrivono un `whoteach_template` e un’architettura trasferibile, ma servono schema reale delle risorse educative, ingestion/chunking, mapping, collection, filtri, prompt e valutazione sul nuovo corpus. Non basta sostituire il prompt.

## Architettura di riferimento (baseline resource-oriented)

```text
Richiesta dell’utente
       │
       ▼
ResourcePlannerAgent ──► piano strutturato (query, filtri, chiarimenti)
       │
       ▼
Resource Workflow (Python: validazioni, stato, trace)
       │
       ▼
MCPResourceToolGateway ──► search_resources / get_resources
       │
       ▼
RetrievalService ──► GTE embeddings ──► Qdrant ──► candidati
       │                                            │
       │                               [reranker opzionale, lato server]
       ▼
ResourceAnswerAgent ──► risposta grounded nei resource_id recuperati
```

Il **Planner non decide** modello di embedding, collection, dimensioni vettoriali o profilo tecnico: questa configurazione è nel servizio/registry lato server. Il reranker è uno scorer esterno a Qdrant e non un agente LLM. Nel profilo `fast` è disabilitato; in `quality` viene applicato ai candidati, con configurazione dichiarata nel JSON MCP.

### File di riferimento citati nei README

| Livello | Sorgenti citate | Responsabilità |
| --- | --- | --- |
| Agenti | `resource_llm_agents.py` | Planner e Answer generici |
| Stato/schemi | `resource_llm_models.py` | Contratti di input/output e trace |
| Orchestrazione lineare | `resource_llm_workflow.py` | Ordine degli step e controllo del flusso |
| Orchestrazione iterativa additiva | `resource_iterative_orchestration.py` | Controller, decisioni, limiti, pause/ripresa |
| Gateway MCP | `resource_mcp_gateway.py` | Invocazione e validazione degli strumenti |
| MCP e servizio | `resource_mcp.py`, `resource_service.py` | Tool pubblici e ricerca applicativa |
| Ricerca vettoriale | `resource_qdrant.py`, `resource_embeddings.py` | Qdrant ed encoder |
| Reranking | `resource_rerankers.py`, `resource_mcp_profiles.py` | Scorer / costruzione profili tecnici |
| Domain adapter | `resource_domain.py`, `resource_adapters.py` | Filtri e mapping dal dominio a risorse generiche |
| Configurazioni | `configs/llm_profiles_v1.json`, `resource_domains_v1.json`, `resource_mcp_profiles_v1.json` | LLM, dominio, profili tecnici |

*I nomi sono ricavati dai documenti: alcuni file potrebbero non essere ancora presenti o avere varianti nel tuo checkout attuale.* Per i dettagli consulta [la mappa tecnica completa](docs/architecture/resource-llm-mcp-workflow.md).

## Come si è arrivati qui

| Fase | Notebook / documenti | Risultato o scopo |
| --- | --- | --- |
| Preparazione dati | `01_loading_visualization_ml-latest-small`, `02_tmdb_enrichment` | Esplorazione MovieLens e arricchimento dei metadati TMDb |
| Valutazione retrieval offline | `03_audit_offline`, `04_retrieval_ablation_50_movies` | Giudizi, metriche e scelta iniziale della baseline dense |
| Primo backend vettoriale | `05_qdrant_dense_integration` | BGE-M3 dense e verifica della corrispondenza del ranking su Qdrant |
| Reranking e MCP specializzato | `06_qdrant_dense_reranking`, `07_mcp_server_validation` | Reranker sperimentale e tool `search_movies` |
| Prima pipeline LLM | `08_llm_mcp_integration` | Planner/Answer per il dominio MovieLens |
| Confronto retrieval esteso | `09_retrieval_ablation_2000_multimodel` | Modelli dense/sparse/fusioni/reranker su corpus maggiore |
| Astrazione del dominio | `10_qdrant_gte_resource_mcp` | GTE, `ResourceDocument`, `search_resources` |
| Workflow generico lineare | `11_resource_llm_mcp_integration_refactored` | Planner → MCP → Answer, stato e trace |
| Estensione iterativa | `orchestration_iterative_v1` | Controller condizionale additivo, con test offline dichiarati |
| Valutazione recommendation | `recommendation_01`, `02`, `04`, `05` | Protocollo cold-start, baseline e benchmark LLM/MCP su filone parallelo |

I riferimenti ai notebook sono **nomi citati nella documentazione**, non una distinta verificata dei file disponibili. L’evoluzione completa è in [PROJECT_EVOLUTION](docs/PROJECT_EVOLUTION.md).

## Esperimenti: filone indipendente dal numero dei notebook di sviluppo

Il protocollo `rs_coldstart_v1` simula diversi livelli di informazioni disponibili all’utente: `k = 0, 1, 3, 5, 10` rating rivelati. I ranking sono valutati con due definizioni della ground truth: **A** (20 rating nascosti fissi) e **B** (tutti i rating non rivelati). La separazione degli input pubblici da `private_evaluation/` è parte del protocollo.

- [Recommendation 01: protocollo](docs/experiments/recommendation-01-coldstart-evaluation-protocol.md) — dataset, split, scenari, valutatore.
- [Recommendation 02: Popularity / MF / GTE](docs/experiments/recommendation-02-popularity-mf-gte-baselines.md) — riferimenti non LLM e selezione degli iperparametri sulla validation.
- [Recommendation 04: LLM → profilo semantico → GTE](docs/experiments/recommendation-04-llm-semantic-profile-gte.md) — confronto appaiato con GTE diretto.
- [Recommendation 05: pipeline LLM/MCP lineare](docs/experiments/recommendation-05-linear-llm-mcp-benchmark.md) — le metriche principali sono sui **risultati del retrieval MCP**, non sul testo dell’Answer.

**Nota sulla numerazione:** nei README consegnati non compare la documentazione di `recommendation_03`; non viene dedotto né rinominato un notebook mancante.

## Esecuzione e controlli prima di procedere

Questo archivio contiene **solo documentazione**, non aggiunge o sostituisce notebook, sorgenti Python, configurazioni o dati. Evita di applicare alla cieca i comandi del [README legacy](docs/history/movielens-bge-m3-legacy-architecture.md) all’architettura più recente.

Per riprendere il lavoro in un checkout aggiornato:

1. Verifica la presenza dei moduli `resource_*` e delle tre configurazioni JSON citate nella guida al [workflow generico](docs/architecture/resource-llm-mcp-workflow.md).
2. Controlla `QDRANT_URL`, la collection GTE effettiva e la coerenza tra document representation, encoder e indice.
3. Riproduci il notebook 10 e quindi il notebook 11, confrontando filtri, ID recuperati, grounding e trace.
4. Per il reranker `quality`, verifica [la configurazione server-side](docs/retrieval/gte-reranker-mcp-profile-refactor.md) e il relativo blocco `reranker` nel JSON.
5. Prima di usare la variante iterativa in live, esegui i test e controlla la compatibilità del suo gateway MCP, come riportato nella [guida all’orchestrazione](docs/orchestration/iterative-orchestration-v1-installation.md).
6. Tratta i notebook `recommendation_*` come un filone sperimentale separato e mantieni validation/test, manifest e cache coerenti.

Non pubblicare `.env`, chiavi API, output che possono contenere richieste personali o cartelle di valutazione privata. I documenti sull’orchestrazione suggeriscono di escludere `outputs/orchestration_runs/` dal versionamento.

## Stato e limiti da non confondere

- **Non c’è evidenza nei soli README di una run live completa** della nuova orchestrazione iterativa sul tuo repository: sono dichiarati smoke test offline/mocks.
- I test sintetici di `recommendation_02`, `04` e `05` **non sono metriche sperimentali reali** su MovieLens.
- I vettori GTE indicizzati potrebbero includere tag e statistiche MovieLens relativi agli utenti di test, con rischio di **information leakage**.
- Le metriche dei due protocolli A/B non sono direttamente confrontabili in valore assoluto quando cambia l’insieme di target rilevanti.
- Per confrontare linear vs iterative serve allineare modelli, prompt di Answer, budget e input: nei documenti la loro generazione della risposta non è ancora identica.
- Il controller iterativo **non** equivale a pianificazione libera di grafi di agenti: usa un ciclo con operazioni delimitate.

Vedi [limiti sperimentali](docs/RESEARCH_LIMITATIONS.md) per i dettagli.

## Nota sul materiale storico

Il file originariamente chiamato semplicemente **`README.md`** è conservato integralmente, ma rinominato come [architettura storica MovieLens](docs/history/movielens-bge-m3-legacy-architecture.md). Non è stato eliminato: le sue configurazioni BGE-M3, i profili `dense_v1` e il contratto `search_movies` servono per capire e riprodurre le prime fasi del lavoro, ma **non** devono essere presentati come la baseline resource-oriented degli ultimi notebook.
