# Indice della documentazione

La documentazione è organizzata **per funzione**, non per ordine d’arrivo dei file. I 10 README originali sono stati mantenuti nei documenti rinominati, con una breve nota iniziale di provenienza e di contesto. Il `README.md` generale, questo indice, `PROJECT_EVOLUTION.md` e `RESEARCH_LIMITATIONS.md` sono i quattro documenti trasversali aggiunti.

[← README generale](../README.md) · [Evoluzione del progetto](PROJECT_EVOLUTION.md) · [Limiti sperimentali](RESEARCH_LIMITATIONS.md)

## 1. Architettura resource-oriented (notebook 10–11)

- [**Resource LLM/MCP — Planner, Retrieval, Answer**](architecture/resource-llm-mcp-workflow.md): guida principale per moduli, agenti, workflow, stato, JSON e porting WhoTeach.
- [**Integrazione GTE/Qdrant/MCP generico**](architecture/gte-qdrant-resource-mcp-integration.md): layer di retrieval con `ResourceDocument`, indice GTE e nuovi tool MCP.
- [**Orchestrazione iterativa v1**](orchestration/iterative-orchestration-v1-installation.md): installazione additiva, switch e limiti del prototipo condizionale.

## 2. Retrieval, setup e configurazione tecnica

- [**Baseline dense BGE-M3 + Qdrant (notebook 05)**](retrieval/bge-m3-qdrant-dense-baseline-v1.md): scelta storica della baseline, collezione e criteri di equivalenza.
- [**Refactor del GTE reranker nei profili MCP**](retrieval/gte-reranker-mcp-profile-refactor.md): registrazione e caricamento lazy da configurazione server-side.

## 3. Esperimenti di raccomandazione

- [**Benchmark cold start v2 — guida attuale**](experiments/recommendation-coldstart-v2.md): protocollo condiviso, bug corretti, selezione B, nuovi embedding e verifiche sintetiche. I quattro documenti seguenti descrivono le versioni precedenti.
- [**01 — Protocollo di valutazione cold-start**](experiments/recommendation-01-coldstart-evaluation-protocol.md): scenari e split; protocolli A/B.
- [**02 — Baseline Popularity / Matrix Factorization / GTE**](experiments/recommendation-02-popularity-mf-gte-baselines.md): confronto semantico e collaborativo.
- [**04 — Profilo semantico LLM + GTE**](experiments/recommendation-04-llm-semantic-profile-gte.md): fase 2A, senza dialogo LLM–LLM.
- [**05 — Benchmark pipeline lineare LLM/MCP**](experiments/recommendation-05-linear-llm-mcp-benchmark.md): analisi dei ranking MCP e Answer separato.

La diagnostica `recommendation_03` corrente è descritta nella guida v2; non era inclusa nei README storici consegnati.

## 4. Storia tecnica

- [**Architettura MovieLens legacy — BGE-M3/Qdrant/MCP/LLM**](history/movielens-bge-m3-legacy-architecture.md): ex `README.md`, notebook 05–09, tool e profili `movie_*` precedenti.
- [**Evoluzione del progetto**](PROJECT_EVOLUTION.md): cronologia e transizione delle interfacce.
- [**Limiti sperimentali e verifiche**](RESEARCH_LIMITATIONS.md): cautele per la tesi e criteri di confronto.

## 5. Mappatura precisa dei nomi originali

| Nome consegnato | Nuovo percorso |
| --- | --- |
| `README.md` | [`docs/history/movielens-bge-m3-legacy-architecture.md`](history/movielens-bge-m3-legacy-architecture.md) |
| `README_QDRANT.md` | [`docs/retrieval/bge-m3-qdrant-dense-baseline-v1.md`](retrieval/bge-m3-qdrant-dense-baseline-v1.md) |
| `INTEGRATION_README.md` | [`docs/architecture/gte-qdrant-resource-mcp-integration.md`](architecture/gte-qdrant-resource-mcp-integration.md) |
| `LLM_RESOURCE_INTEGRATION_README(1).md` | [`docs/architecture/resource-llm-mcp-workflow.md`](architecture/resource-llm-mcp-workflow.md) |
| `RERANKER_CONFIG_REFACTOR(1).md` | [`docs/retrieval/gte-reranker-mcp-profile-refactor.md`](retrieval/gte-reranker-mcp-profile-refactor.md) |
| `README_INSTALLAZIONE(1).md` | [`docs/orchestration/iterative-orchestration-v1-installation.md`](orchestration/iterative-orchestration-v1-installation.md) |
| `README_recommendation_protocol(1).md` | [`docs/experiments/recommendation-01-coldstart-evaluation-protocol.md`](experiments/recommendation-01-coldstart-evaluation-protocol.md) |
| `README_recommendation_02.md` | [`docs/experiments/recommendation-02-popularity-mf-gte-baselines.md`](experiments/recommendation-02-popularity-mf-gte-baselines.md) |
| `README_recommendation_04_fase_2A(1).md` | [`docs/experiments/recommendation-04-llm-semantic-profile-gte.md`](experiments/recommendation-04-llm-semantic-profile-gte.md) |
| `README_recommendation_05_pipeline_lineare_benchmark.md` | [`docs/experiments/recommendation-05-linear-llm-mcp-benchmark.md`](experiments/recommendation-05-linear-llm-mcp-benchmark.md) |

I documenti storici mantengono il proprio contesto e non sono stati riscritti come se descrivessero lo stato attuale. I numeri dei notebook di sviluppo (`01`–`11`) sono **un filone distinto** da quelli del protocollo `recommendation_01`, `_02`, `_04`, `_05`.
