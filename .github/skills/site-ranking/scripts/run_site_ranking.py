from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import requests


_CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"
_OPENALEX_INST_URL = "https://api.openalex.org/institutions"


def _load_payload(json_arg: str | None, file_arg: str | None) -> dict[str, Any]:
    if bool(json_arg) == bool(file_arg):
        raise SystemExit("Provide exactly one of --json or --file")
    if json_arg:
        return json.loads(json_arg)
    return json.loads(Path(file_arg).read_text(encoding="utf-8"))


def _calculate_centrality(G: nx.Graph, metric: str) -> dict[str, float]:
    metric = (metric or "degree").lower()
    if metric == "degree":
        return nx.degree_centrality(G)
    if metric == "betweenness":
        return nx.betweenness_centrality(G)
    if metric == "closeness":
        return nx.closeness_centrality(G)
    if metric == "eigenvector":
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            return nx.degree_centrality(G)
    if metric == "pagerank":
        return nx.pagerank(G)
    return nx.degree_centrality(G)


async def _search_clinical_trials(params_in: dict[str, Any]) -> dict[str, Any]:
    params_in = params_in or {}
    condition = str(params_in.get("condition", "") or "")
    intervention = str(params_in.get("intervention", "") or "")
    location = str(params_in.get("location", "") or "")
    other_terms = str(params_in.get("other_terms", "") or "")
    sponsor = str(params_in.get("sponsor", "") or "")
    status = str(params_in.get("status", "") or "")

    max_results = int(params_in.get("max_results", 100) or 100)
    max_results = max(1, min(max_results, 100))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://clinicaltrials.gov/",
        "DNT": "1",
    }

    params: dict[str, Any] = {"pageSize": max_results, "format": "json"}
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if location:
        params["query.locn"] = location
    if sponsor:
        params["query.spons"] = sponsor
    if other_terms:
        params["query.term"] = other_terms
    if status:
        params["filter.overallStatus"] = status

    def _do_request() -> dict[str, Any]:
        resp = requests.get(_CT_API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    try:
        data = await asyncio.to_thread(_do_request)
    except Exception as e:
        return {
            "error": str(e),
            "search_params": {
                "condition": condition,
                "intervention": intervention,
                "location": location,
                "other_terms": other_terms,
                "sponsor": sponsor,
                "status": status,
                "max_results": max_results,
            },
        }

    studies_out: list[dict[str, Any]] = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        contacts = protocol.get("contactsLocationsModule", {})
        references_module = protocol.get("referencesModule", {})

        overall_officials: list[dict[str, Any]] = []
        for official in contacts.get("overallOfficials", []) or []:
            overall_officials.append(
                {
                    "name": official.get("name"),
                    "affiliation": official.get("affiliation"),
                    "role": official.get("role"),
                }
            )

        sites: list[dict[str, Any]] = []
        for loc in contacts.get("locations", []) or []:
            sites.append(
                {
                    "facility": loc.get("facility"),
                    "city": loc.get("city"),
                    "state": loc.get("state"),
                    "country": loc.get("country"),
                    "status": loc.get("status"),
                    "contacts": loc.get("contacts", []) or [],
                }
            )

        publications: list[dict[str, Any]] = []
        for ref in references_module.get("references", []) or []:
            if "pmid" in ref:
                publications.append(
                    {"pmid": ref.get("pmid"), "type": ref.get("type"), "citation": ref.get("citation")}
                )

        studies_out.append(
            {
                "nct_id": identification.get("nctId"),
                "title": identification.get("briefTitle"),
                "status": status_module.get("overallStatus"),
                "phase": protocol.get("designModule", {}).get("phases", []) or [],
                "sponsor": sponsor_module.get("leadSponsor", {}).get("name"),
                "conditions": protocol.get("conditionsModule", {}).get("conditions", []) or [],
                "sites": sites,
                "enrollment": status_module.get("enrollmentInfo", {}) or {},
                "start_date": status_module.get("startDateStruct", {}) or {},
                "publications": publications,
                "overall_officials": overall_officials,
            }
        )

    return {
        "total_count": len(studies_out),
        "studies": studies_out,
        "retrieved_at": datetime.now().isoformat(),
        "search_params": {
            "condition": condition,
            "intervention": intervention,
            "location": location,
            "other_terms": other_terms,
            "sponsor": sponsor,
            "status": status,
            "max_results": max_results,
        },
    }


async def _standardize_sites_openalex(
    trial_data: dict[str, Any],
    *,
    email_for_api: str = "",
    max_matches: int = 5,
    concurrency: int = 20,
) -> dict[str, Any]:
    studies = trial_data.get("studies", []) or []

    unique_facilities: set[str] = set()
    for st in studies:
        for site in st.get("sites", []) or []:
            fac = str(site.get("facility") or "").strip()
            if fac:
                unique_facilities.add(fac)

    if not unique_facilities:
        return trial_data

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    cache: dict[str, dict[str, Any]] = {}

    async def _fetch_one(facility: str) -> tuple[str, dict[str, Any]]:
        async with sem:
            params = {"search": facility, "per_page": int(max_matches), "mailto": email_for_api}

            def _do_request() -> dict[str, Any]:
                resp = requests.get(_OPENALEX_INST_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.json()

            try:
                data = await asyncio.to_thread(_do_request)
                matches = data.get("results", []) or []
                if matches:
                    best = matches[0]
                    std = {
                        "standardized_name": best.get("display_name") or facility,
                        "openalex_id": best.get("id"),
                        "ror": best.get("ror"),
                        "country_code": best.get("country_code"),
                    }
                else:
                    std = {"standardized_name": facility, "openalex_id": None, "ror": None, "country_code": None}
            except Exception:
                std = {"standardized_name": facility, "openalex_id": None, "ror": None, "country_code": None}

            return facility, std

    facilities = sorted(unique_facilities)
    results = await asyncio.gather(*[_fetch_one(f) for f in facilities])
    for facility, std in results:
        cache[facility] = std

    for st in studies:
        for site in st.get("sites", []) or []:
            fac = str(site.get("facility") or "").strip()
            if fac and fac in cache:
                std = cache[fac]
                site["standardized_facility"] = std["standardized_name"]
                site["openalex_id"] = std["openalex_id"]
                site["ror"] = std["ror"]
                site["original_facility"] = fac

    trial_data["standardization"] = {
        "openalex": True,
        "unique_facilities": len(unique_facilities),
        "standardized_at": datetime.now().isoformat(),
    }
    return trial_data


def _rank_sites(
    trial_data: dict[str, Any],
    *,
    metric: str,
    top_n: int,
    filters: dict[str, Any] | None,
) -> dict[str, Any]:
    filters = filters or {}
    studies = (trial_data or {}).get("studies", []) or []
    if not studies:
        return {"error": "No studies found in data"}

    G = nx.Graph()
    site_metadata: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trials": set(), "conditions": set(), "phases": set()}
    )

    for study in studies:
        nct_id = study.get("nct_id")
        conditions = study.get("conditions", []) or []
        phases = study.get("phase", []) or []
        sites = study.get("sites", []) or []

        site_ids: list[str] = []
        for site in sites:
            facility = site.get("standardized_facility") or site.get("facility") or "Unknown"
            original_facility = site.get("original_facility", facility)
            city = site.get("city", "") or ""
            state = site.get("state", "") or ""
            country = site.get("country", "") or ""
            openalex_id = site.get("openalex_id")

            site_id = f"{facility}|{city}|{state}|{country}"
            site_ids.append(site_id)

            if not G.has_node(site_id):
                G.add_node(
                    site_id,
                    facility=facility,
                    original_facility=original_facility,
                    city=city,
                    state=state,
                    country=country,
                    openalex_id=openalex_id,
                )

            site_metadata[site_id]["trials"].add(nct_id)
            site_metadata[site_id]["conditions"].update(conditions)
            site_metadata[site_id]["phases"].update(phases)

        for i, s1 in enumerate(site_ids):
            for s2 in site_ids[i + 1 :]:
                if G.has_edge(s1, s2):
                    G[s1][s2]["weight"] += 1
                else:
                    G.add_edge(s1, s2, weight=1)

    if G.number_of_nodes() == 0:
        return {"error": "No sites found to rank"}

    metric_norm = (metric or "degree").lower()
    if metric_norm == "frequency":
        ranking = {sid: len(site_metadata[sid]["trials"]) for sid in G.nodes()}
    else:
        ranking = _calculate_centrality(G, metric_norm)

    ranked_sites = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if filters:
        for fk, fv in filters.items():
            if fv:
                ranked_sites = [
                    (sid, score)
                    for sid, score in ranked_sites
                    if str(G.nodes[sid].get(str(fk), "")).lower() == str(fv).lower()
                ]

    ranked_sites = ranked_sites[: int(top_n)]

    results: list[dict[str, Any]] = []
    for sid, score in ranked_sites:
        node = G.nodes[sid]
        meta = site_metadata[sid]
        results.append(
            {
                "rank": len(results) + 1,
                "facility": node.get("facility"),
                "original_facility": node.get("original_facility"),
                "location": f"{node.get('city','')}, {node.get('state','')}, {node.get('country','')}",
                "centrality_score": round(float(score), 4),
                "num_trials": len(meta["trials"]),
                "num_collaborations": G.degree(sid),
                "conditions": list(meta["conditions"])[:5],
                "phases": list(meta["phases"]),
                "openalex_id": node.get("openalex_id"),
            }
        )

    out: dict[str, Any] = {
        "metric": metric_norm,
        "top_sites": results,
        "network_stats": {
            "total_sites": G.number_of_nodes(),
            "total_connections": G.number_of_edges(),
            "average_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2)
            if G.number_of_nodes() > 0
            else 0,
            "network_density": round(nx.density(G), 4),
        },
        "analyzed_at": datetime.now().isoformat(),
    }
    if filters:
        out["filters_applied"] = filters
    return out


async def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload or {}

    metric = str(payload.get("metric") or payload.get("ranking", {}).get("metric") or "degree")
    top_n = int(payload.get("top_n") or payload.get("ranking", {}).get("top_n") or 10)

    filters = payload.get("filters")
    if filters is None:
        filters = payload.get("ranking", {}).get("filters", {})
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object")

    standardize = payload.get("standardize_openalex")
    if standardize is None:
        standardize = payload.get("ranking", {}).get("standardize_openalex", True)
    standardize = bool(standardize)

    trial_data = payload.get("trial_data")
    search_params = payload.get("search")

    if bool(trial_data) == bool(search_params):
        raise ValueError("Provide exactly one of trial_data or search")

    if search_params is not None:
        if not isinstance(search_params, dict):
            raise ValueError("search must be an object")
        trial_data = await _search_clinical_trials(search_params)

    if not isinstance(trial_data, dict):
        raise ValueError("trial_data must be an object")

    if trial_data.get("error"):
        return {"error": trial_data["error"], "trial_data": trial_data}

    if standardize:
        email = str(payload.get("email_for_api") or os.getenv("EMAIL_FOR_API", "") or "")
        trial_data = await _standardize_sites_openalex(trial_data, email_for_api=email)

    ranking = _rank_sites(trial_data, metric=metric, top_n=top_n, filters=filters)

    return {
        "search": trial_data.get("search_params"),
        "standardize_openalex": standardize,
        "ranking": ranking,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ClinicalTrials.gov site ranking (search + optional OpenAlex + rank)")
    ap.add_argument("--json", help="Payload as a JSON string")
    ap.add_argument("--file", help="Path to a JSON file containing the payload")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = ap.parse_args()

    payload = _load_payload(args.json, args.file)
    result = asyncio.run(run_from_payload(payload))

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
