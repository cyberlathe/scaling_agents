"""tools/mock_web_search.py — mock web search returning realistic prospect data."""

RESULTS_DB = {
    "fleetsync": [{
        "title": "FleetSync AI raises $28M Series B for last-mile delivery platform",
        "snippet": "FleetSync AI, a logistics intelligence startup, announced a $28M Series B. "
                   "200 employees. Serves 40 enterprise clients. CTO Alex Chen previously led "
                   "engineering at Flexport.",
        "url": "https://techcrunch.com/fleetsync-series-b",
        "source": "TechCrunch, July 2026",
    }],
    "logipath": [{
        "title": "LogiPath records 3× YoY growth in Q2 2026",
        "snippet": "LogiPath recorded strong growth driven by expansion into new markets. "
                   "VP Operations Sam Rivera cited route optimisation as the key bottleneck.",
        "url": "https://logistics-weekly.com/logipath-q2",
        "source": "Logistics Weekly, June 2026",
    }],
    "cargonest": [{
        "title": "CargoNest closes $1.4M seed from Accel",
        "snippet": "CargoNest is building AI-native logistics infrastructure for D2C brands. "
                   "30-person team. CEO Jordan Kim previously led ops at a major 3PL.",
        "url": "https://techcrunch.com/cargonest-seed",
        "source": "TechCrunch, May 2026",
    }],
    "bluecore": [{
        "title": "Bluecore raises $50M for retail marketing personalisation platform",
        "snippet": "Bluecore, a retail marketing technology company, helps brands personalise "
                   "email campaigns using behavioural data. Focus: retail, not logistics.",
        "url": "https://venturebeat.com/bluecore-funding",
        "source": "VentureBeat, 2025",
    }],
    "logistics saas": [{
        "title": "Top logistics SaaS companies raising in 2026",
        "snippet": "FleetSync AI, LogiPath, and CargoNest are among the fastest-growing "
                   "logistics SaaS companies with recent funding rounds.",
        "url": "https://logistics-weekly.com/saas-2026",
        "source": "Logistics Weekly, Aug 2026",
    }],
}


def search_web(query: str) -> dict:
    q = query.lower()
    for key, results in RESULTS_DB.items():
        if key in q:
            return {"results": results, "query": query, "count": len(results)}
    # Generic fallback
    return {
        "results": [{
            "title": f"Search results for: {query}",
            "snippet": "Multiple companies identified as potential targets.",
            "url": "https://search.example.com",
            "source": "Web search",
        }],
        "query": query, "count": 1,
    }
