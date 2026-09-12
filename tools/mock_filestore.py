"""tools/mock_filestore.py — mock file storage."""
import copy
from datetime import datetime

_FILES = {
    "/legal/agreements/FleetTech_NDA_2024.pdf": {
        "path": "/legal/agreements/FleetTech_NDA_2024.pdf",
        "type": "NDA", "status": "active", "company": "FleetTech",
        "signed": "2024-03-15", "expires": "2027-03-15",
        "summary": (
            "Mutual NDA between Company and FleetTech. Covers all commercial discussions. "
            "Governing law: Delaware. Term: 3 years. Auto-renews unless terminated in writing."
        ),
    },
    "/legal/proposals/DispatchPro_proposal_v2.pdf": {
        "path": "/legal/proposals/DispatchPro_proposal_v2.pdf",
        "type": "Proposal", "status": "closed", "company": "DispatchPro",
        "signed": None, "expires": None,
        "summary": "Sales proposal for DispatchPro. Deal closed-lost May 2026.",
    },
    "/templates/outreach_logistics.docx": {
        "path": "/templates/outreach_logistics.docx",
        "type": "Template", "status": "active", "company": None,
        "summary": "Standard outreach email template for logistics vertical.",
    },
}

_store = copy.deepcopy(_FILES)


def list_files(query: str = "", folder: str = "") -> dict:
    results = [
        m for m in _store.values()
        if (not folder or m["path"].startswith(folder))
        and (not query or query.lower() in m["path"].lower()
             or query.lower() in (m.get("company") or "").lower())
    ]
    return {"count": len(results), "files": [{k: v for k, v in f.items() if k != "summary"} for f in results]}


def read_file(path: str) -> dict:
    if path not in _store:
        return {"success": False, "error": f"File not found: {path}"}
    m = _store[path]
    return {
        "success": True, "path": path, "type": m["type"],
        "status": m["status"], "company": m.get("company"),
        "signed": m.get("signed"), "expires": m.get("expires"),
        "content_summary": m["summary"],
    }
