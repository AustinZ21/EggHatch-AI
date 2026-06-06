# Graph Report - EggHatch-AI  (2026-06-06)

## Corpus Check
- 16 files · ~82,187 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 190 nodes · 232 edges · 15 communities (10 shown, 5 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2415576a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `DataPipeline` - 18 edges
2. `TrendAnalysisAgent` - 17 edges
3. `build_laptop_comparison()` - 13 edges
4. `OllamaClient` - 8 edges
5. `SentimentAnalyzer` - 8 edges
6. `OllamaResponse` - 6 edges
7. `_to_float()` - 6 edges
8. `AgentState` - 5 edges
9. `get_data_pipeline()` - 5 edges
10. `get_sentiment_analyzer()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `test_builds_explainable_comparison_report()` --calls--> `build_laptop_comparison()`  [INFERRED]
  tests/test_comparison.py → app/agents/comparison.py
- `test_budget_filter_keeps_engineered_candidate_fields()` --calls--> `build_laptop_comparison()`  [INFERRED]
  tests/test_comparison.py → app/agents/comparison.py
- `test_data_pipeline_loads_fixture_data()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py
- `test_data_pipeline_filters_reviews_by_query()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py
- `test_preprocess_data_adds_recommendation_features()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py

## Communities (15 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (34): ask_for_budget(), ask_for_use_case(), check_information_completeness(), collect_streaming_response(), create_agent_graph(), decompose_tasks(), extract_task_queue(), initialize_state() (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (23): Simple rule-based sentiment analyzer as fallback., Simple rule-based sentiment analyzer as fallback., Initialize and train the LDA topic model., Initialize and train the LDA topic model., Analyze sentiment for each topic., Analyze sentiment for each topic., Analyze trends and sentiment in gaming laptop reviews.                  Args:, Analyze trends and sentiment in gaming laptop reviews.                  Args: (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (19): DataPipeline, get_data_pipeline(), Data Pipeline Module for EggHatch AI.  This module handles data loading, prepr, Clean and normalize text., Get cleaned review texts for topic modeling., Get processed reviews with metadata., Get laptop data as DataFrame., Filter reviews by a search query. (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (24): build_laptop_comparison(), _candidate_metrics(), _cautions(), _comparison_mode(), _dimension_reason(), looks_like_comparison_query(), _normalize(), Deterministic laptop comparison helpers for EggHatch-AI.  This module keeps comp (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (13): OllamaClient, OllamaResponse, LLM Integration module for EggHatch AI.  This module provides client wrappers, Generate a chat completion using the Ollama API.                  Args:, Stream responses from the Ollama API.                  Args:             endp, Schema for Ollama API response., Client for interacting with Ollama API., Initialize the Ollama client.                  Args:             base_url: Ba (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (11): get_sentiment_analyzer(), Sentiment Analysis Module for EggHatch AI.  This module specializes in sentime, Analyze sentiment for a list of texts.                  Args:             tex, Get overall sentiment overview for all reviews.                  Args:, Get a sentiment analyzer instance., Analyzes sentiment in tech product reviews., Initialize the sentiment analysis model., Simple rule-based sentiment analyzer as fallback.                  Args: (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (5): Trend Analysis Module for EggHatch AI.  This module focuses on topic modeling, Run the trend analysis agent and update the state.          Args:         sta, Run the trend analysis agent and update the state.          Args:         sta, State, trend_analysis_agent()

### Community 7 - "Community 7"
Cohesion: 0.5
Nodes (3): Streamlit dashboard for EggHatch AI.  This module implements a user-friendly i, Render a structured laptop comparison when available., render_comparison_block()

## Knowledge Gaps
- **95 isolated node(s):** `Streamlit dashboard for EggHatch AI.  This module implements a user-friendly i`, `Render a structured laptop comparison when available.`, `LLM Integration module for EggHatch AI.  This module provides client wrappers`, `Schema for Ollama API response.`, `Client for interacting with Ollama API.` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrendAnalysisAgent` connect `Community 1` to `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.511) - this node is a cross-community bridge._
- **Why does `get_data_pipeline()` connect `Community 2` to `Community 1`, `Community 5`?**
  _High betweenness centrality (0.258) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `DataPipeline` (e.g. with `test_data_pipeline_loads_fixture_data()` and `test_data_pipeline_filters_reviews_by_query()`) actually correct?**
  _`DataPipeline` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `TrendAnalysisAgent` (e.g. with `AgentState` and `execute_current_task()`) actually correct?**
  _`TrendAnalysisAgent` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `build_laptop_comparison()` (e.g. with `.analyze_trends()` and `test_builds_explainable_comparison_report()`) actually correct?**
  _`build_laptop_comparison()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Streamlit dashboard for EggHatch AI.  This module implements a user-friendly i`, `Render a structured laptop comparison when available.`, `LLM Integration module for EggHatch AI.  This module provides client wrappers` to the rest of the system?**
  _95 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._