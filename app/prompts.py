"""
Prompt templates for EggHatch AI - POC Level.

This module contains simplified prompt templates for the EggHatch AI system,
optimized for Gemma 3 12B.

Note: For the POC, only the trend analysis module is fully implemented.
The product knowledge, pricing availability, and build recommendation modules
are planned for future development.
"""

# Master Agent / Orchestrator prompts
MASTER_AGENT_SYSTEM_PROMPT = """
You are EggHatch AI, a PC building and tech gear shopping assistant.
Your task is to help users with PC components, gaming laptops, and tech gear recommendations.
Focus on understanding their needs, budget, and preferences to provide helpful suggestions.
"""

QUERY_UNDERSTANDING_PROMPT = """
Analyze this query about PC components or tech gear:

USER QUERY: {user_query}

Extract and return a JSON object with:
- query_type: Type of query (gaming_pc_build, laptop_recommendation, component_advice)
- budget: Budget mentioned (or null)
- use_case: Intended use (gaming, content_creation, office_work)
- requirements: Specific requirements or preferences

Format: ```json
{{
  "query_type": "example",
  "budget": "$1000",
  "use_case": "example",
  "requirements": {{}}
}}
```
"""

# Specialized Tool prompts

# Current implementation
TREND_ANALYSIS_PROMPT = """
Analyze trends and sentiment for: {products_or_categories}

Based on the data, provide insights on:
1. Popular topics and features identified through topic modeling
2. Sentiment analysis results (positive/negative/neutral classifications)
3. Key praises/complaints extracted from reviews
4. Feature popularity and associated sentiment scores
5. Recommendations based on combined trend and sentiment analysis

Use both topic modeling results and sentiment analysis scores to provide comprehensive insights.

Data: {trend_data}
"""

# Response Synthesis prompt
RESPONSE_SYNTHESIS_PROMPT = """
Create a concise, helpful response for this query: {user_query}

Use ONLY the following real data from our analysis of 250+ gaming laptops and reviews:
- Trends and Sentiment Analysis: {trend_insights}

Your response MUST:
1. Be brief and to the point (max 150 words)
2. Provide 2-3 SPECIFIC laptop recommendations from our dataset with ACTUAL prices
3. Reference REAL features and sentiment scores from the data provided
4. NOT include ANY meta-commentary about the analysis process
5. NOT mention hypothetical data or placeholder information

The user wants factual recommendations based on our actual laptop database, not generic advice.
"""
