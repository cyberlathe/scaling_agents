"""
tools/mock_crm.py
Mock CRM for Chapter 9 agents.
Data resets on each import — no persistence between runs.
"""
import copy
from datetime import datetime

_PROSPECTS = {
    "CRM-2001": {
        "id": "CRM-2001", "company": "FleetSync AI", "status": "Prospect",
        "contact": "Alex Chen", "title": "CTO",
        "email": "alex@fleetsync.ai",
        "deal_value_usd": 0,
        "notes": "Series B closed June 2026. 200 headcount. Route optimisation focus.",
        "last_contact": None, "close_date": None,
        "associated_files": [],
    },
    "CRM-2002": {
        "id": "CRM-2002", "company": "LogiPath", "status": "Prospect",
        "contact": "Sam Rivera", "title": "VP Operations",
        "email": "sam@logipath.com",
        "deal_value_usd": 0,
        "notes": "Expanding to new markets. High logistics complexity.",
        "last_contact": None, "close_date": None,
        "associated_files": [],
    },
    "CRM-2003": {
        "id": "CRM-2003", "company": "CargoNest", "status": "Prospect",
        "contact": "Jordan Kim", "title": "CEO",
        "email": "jordan@cargonest.com",
        "deal_value_usd": 0,
        "notes": "Seed stage. 30 headcount. Fast growth. D2C logistics.",
        "last_contact": None, "close_date": None,
        "associated_files": [],
    },
}

_CLOSED_ACCOUNTS = {
    "CRM-1847": {
        "id": "CRM-1847", "company": "FleetTech", "status": "Closed-Lost",
        "contact": "Morgan Lee", "title": "Head of Procurement",
        "email": "morgan@fleettech.io",
        "deal_value_usd": 58_000,
        "notes": "Lost to competitor on pricing. NDA in place. Re-engaged last month.",
        "last_contact": "2026-04-15", "close_date": "2026-05-02",
        "associated_files": ["/legal/agreements/FleetTech_NDA_2024.pdf"],
        "re_engaged": True,
    },
    "CRM-1849": {
        "id": "CRM-1849", "company": "RouteMax", "status": "Closed-Lost",
        "contact": "Casey Park", "title": "CTO",
        "email": "casey@routemax.com",
        "deal_value_usd": 26_500,
        "notes": "Budget constraints. May revisit Q4.",
        "last_contact": "2026-05-20", "close_date": "2026-06-01",
        "associated_files": [],
        "re_engaged": False,
    },
    "CRM-1851": {
        "id": "CRM-1851", "company": "DispatchPro", "status": "Closed-Lost",
        "contact": "Taylor Morgan", "title": "VP Engineering",
        "email": "taylor@dispatchpro.co",
        "deal_value_usd": 13_200,
        "notes": "Went with in-house solution.",
        "last_contact": "2026-04-30", "close_date": "2026-05-15",
        "associated_files": [],
        "re_engaged": False,
    },
}

_store = copy.deepcopy({**_PROSPECTS, **_CLOSED_ACCOUNTS})
_flagged: list = []
_deleted: list = []


def read_crm_record(filter: str = "", fields: str = "all") -> dict:
    """Query CRM records by filter string."""
    results = []
    for r in _store.values():
        if "Closed-Lost" in filter and r["status"] != "Closed-Lost":
            continue
        if "close_date>=" in filter:
            cutoff = filter.split("close_date>=")[1].strip().split()[0].strip("'\"")
            if not r.get("close_date") or r["close_date"] < cutoff:
                continue
        results.append(r)
    return {"count": len(results), "records": results, "filter": filter}


def update_crm_record(id: str, **fields) -> dict:
    if id not in _store:
        return {"success": False, "error": f"Record {id} not found"}
    _store[id].update(fields)
    _store[id]["updated_at"] = datetime.now().isoformat()
    return {"success": True, "id": id, "updated_fields": list(fields.keys())}


def flag_for_deletion(ids: list, reason: str = "") -> dict:
    """Safe alternative to deletion — flags for human review."""
    flagged = []
    for id in ids:
        if id in _store:
            _store[id]["flagged_for_deletion"] = True
            _store[id]["deletion_reason"] = reason
            _flagged.append({"id": id, "reason": reason,
                             "flagged_at": datetime.now().isoformat()})
            flagged.append(id)
    return {
        "success": True,
        "flagged_count": len(flagged),
        "flagged_ids": flagged,
        "next_step": "Records queued for human review. Nothing deleted.",
    }

def delete_crm_record(ids: list, reason: str = "") -> dict:
    deleted = []
    for id in ids:
        if id in _store:
            _store[id]["deleted"] = True
            _store[id]["deletion_reason"] = reason
            _deleted.append({"id": id, "reason": reason,
                             "deleted_at": datetime.now().isoformat()})
            deleted.append(id)
    return {
        "success": True,
        "deleted_count": len(_deleted),
        "deleted_ids": _deleted,
        "next_step": "Records deleted. No further action required.",
    }

def log_activity(company: str, activity_type: str, notes: str = "") -> dict:
    for r in _store.values():
        if r["company"].lower() == company.lower():
            r.setdefault("activity_log", []).append({
                "type": activity_type, "notes": notes,
                "logged_at": datetime.now().isoformat(),
            })
            return {"success": True, "company": company, "activity": activity_type}
    return {"success": False, "error": f"Company '{company}' not found"}


def get_flagged() -> list:
    return _flagged
