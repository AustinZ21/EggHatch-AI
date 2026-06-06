# Graph Report - EggHatch-AI  (2026-06-06)

## Corpus Check
- 14 files · ~80,177 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 143 nodes · 171 edges · 14 communities (8 shown, 6 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 10 edges (avg confidence: 0.74)
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
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `DataPipeline` - 18 edges
2. `TrendAnalysisAgent` - 16 edges
3. `OllamaClient` - 8 edges
4. `SentimentAnalyzer` - 8 edges
5. `OllamaResponse` - 6 edges
6. `AgentState` - 5 edges
7. `get_data_pipeline()` - 5 edges
8. `get_sentiment_analyzer()` - 5 edges
9. `collect_streaming_response()` - 4 edges
10. `execute_current_task()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_data_pipeline_loads_fixture_data()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py
- `test_data_pipeline_filters_reviews_by_query()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py
- `test_preprocess_data_adds_recommendation_features()` --calls--> `DataPipeline`  [INFERRED]
  tests/test_data_pipeline.py → app/agents/data_pipeline.py
- `AgentState` --uses--> `TrendAnalysisAgent`  [INFERRED]
  app/master_agent.py → app/agents/trend_analysis.py
- `execute_current_task()` --calls--> `TrendAnalysisAgent`  [INFERRED]
  app/master_agent.py → app/agents/trend_analysis.py

## Communities (14 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (16): Trend Analysis Module for EggHatch AI.  This module focuses on topic modeling, Simple rule-based sentiment analyzer as fallback., Initialize and train the LDA topic model., Analyze sentiment for each topic., Analyze trends and sentiment in gaming laptop reviews.                  Args:, Identify popular features mentioned in reviews using NLP techniques., Fallback method to identify features using keyword matching., Agent for analyzing trends and feature popularity in tech product reviews. (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (16): DataPipeline, Clean and normalize text., Get cleaned review texts for topic modeling., Get processed reviews with metadata., Get laptop data as DataFrame., Filter reviews by a search query., Preprocess laptop data for modeling., Get feature vectors ready for modeling. (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (27): ask_for_budget(), ask_for_use_case(), check_information_completeness(), collect_streaming_response(), create_agent_graph(), decompose_tasks(), extract_task_queue(), initialize_state() (+19 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (13): OllamaClient, OllamaResponse, LLM Integration module for EggHatch AI.  This module provides client wrappers, Generate a chat completion using the Ollama API.                  Args:, Stream responses from the Ollama API.                  Args:             endp, Schema for Ollama API response., Client for interacting with Ollama API., Initialize the Ollama client.                  Args:             base_url: Ba (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (11): get_sentiment_analyzer(), Sentiment Analysis Module for EggHatch AI.  This module specializes in sentime, Analyze sentiment for a list of texts.                  Args:             tex, Get overall sentiment overview for all reviews.                  Args:, Get a sentiment analyzer instance., Analyzes sentiment in tech product reviews., Initialize the sentiment analysis model., Simple rule-based sentiment analyzer as fallback.                  Args: (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.5
Nodes (3): get_data_pipeline(), Data Pipeline Module for EggHatch AI.  This module handles data loading, prepr, Get a data pipeline instance.

## Knowledge Gaps
- **64 isolated node(s):** `Streamlit dashboard for EggHatch AI.  This module implements a user-friendly i`, `LLM Integration module for EggHatch AI.  This module provides client wrappers`, `Schema for Ollama API response.`, `Client for interacting with Ollama API.`, `Initialize the Ollama client.                  Args:             base_url: Ba` (+59 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrendAnalysisAgent` connect `Community 0` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.409) - this node is a cross-community bridge._
- **Why does `get_data_pipeline()` connect `Community 5` to `Community 0`, `Community 1`, `Community 4`?**
  _High betweenness centrality (0.308) - this node is a cross-community bridge._
- **Why does `DataPipeline` connect `Community 1` to `Community 5`?**
  _High betweenness centrality (0.305) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `DataPipeline` (e.g. with `test_data_pipeline_loads_fixture_data()` and `test_data_pipeline_filters_reviews_by_query()`) actually correct?**
  _`DataPipeline` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `TrendAnalysisAgent` (e.g. with `AgentState` and `execute_current_task()`) actually correct?**
  _`TrendAnalysisAgent` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Streamlit dashboard for EggHatch AI.  This module implements a user-friendly i`, `LLM Integration module for EggHatch AI.  This module provides client wrappers`, `Schema for Ollama API response.` to the rest of the system?**
  _64 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._