# Site-ranking prompt conventions

## Location vs filters
- **Trials in a location** ⇒ use `search.location`.
- **Top sites in a location** ⇒ search broadly (empty `search.location`), then apply `ranking.filters` after ranking.

## Metric selection
- `frequency`: “most trials” / sponsor footprint.
- `degree` / `pagerank`: collaboration hubs/influence.
- `betweenness`: bridge sites that connect clusters.

## Search size
- Prefer `max_results >= 100` when ranking.
