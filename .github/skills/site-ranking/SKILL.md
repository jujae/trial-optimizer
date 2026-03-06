---
name: site-ranking
description: Rank ClinicalTrials.gov clinical trial sites (facilities) for a condition/sponsor/topic using a deterministic Python script (search + optional OpenAlex standardization + NetworkX centrality/frequency + post-ranking geographic filters). Use this skill whenever the user asks for “top sites”, “best sites”, “site ranking”, “hub sites”, “centrality/PageRank of sites”, or wants site recommendations by geography (country/state/city) for clinical trials.
---

# Site Ranking (standalone)

This skill provides a deterministic implementation of ds_ai_platform-style **site ranking**:

1) Search **ClinicalTrials.gov API v2** for studies.
2) (Optional, default) Standardize facility names via **OpenAlex** to help deduplicate.
3) Build a **site co-participation network** (two sites are connected if they appear in the same trial).
4) Rank sites by either:
   - `frequency` (most trials)
   - network centrality (`degree`, `pagerank`, `betweenness`, `closeness`, `eigenvector`)
5) Apply optional **filters after ranking** (country/state/city/etc.).

## Dependencies
Install on your server:

```bash
pip install -r .github/skills/site-ranking/requirements.txt
```

## Prompt conventions (critical)
### Location vs filters
- If the user asks for trials **in** a location (e.g., “trials in California”), put that in `search.location`.
- If the user asks for **top sites in** a location (e.g., “top US sites for breast cancer”), do **NOT** constrain the initial search with `search.location`; instead apply `ranking.filters` after ranking (e.g., `{ "country": "United States" }`).

Rationale: centrality rankings are more informative with a broader network; filtering is safer after ranking.

### Metric choice
- Use `frequency` for “most trials” / footprint.
- Use `degree` (default) or `pagerank` for collaboration hubs / influence.

## How to run
1) Create a payload file `payload.json`.
2) Run:

```bash
python .github/skills/site-ranking/scripts/run_site_ranking.py --file payload.json --pretty
```

## Payload schema
Provide **exactly one** of:
- `search`: run full pipeline (search → optional standardize → rank)
- `trial_data`: skip search and rank the provided ClinicalTrials-like JSON

Fields:
- `search`: object with keys `condition`, `intervention`, `other_terms`, `sponsor`, `status`, `location`, `max_results`
- `ranking`: object with keys:
  - `metric`: `frequency|degree|betweenness|closeness|eigenvector|pagerank`
  - `top_n`: integer
  - `filters`: object (applied after ranking). Common keys: `country`, `state`, `city`, `facility`, `openalex_id`.
  - `standardize_openalex`: boolean (default true)

Optional:
- `email_for_api`: passed as OpenAlex `mailto` (or set env var `EMAIL_FOR_API`).

## Example payload
```json
{
  "search": {
    "condition": "breast cancer",
    "other_terms": "AREA[Phase]Phase 3",
    "status": "RECRUITING",
    "location": "",
    "max_results": 100
  },
  "ranking": {
    "metric": "degree",
    "top_n": 10,
    "filters": {"country": "United States"},
    "standardize_openalex": true
  }
}
```
