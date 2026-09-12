"""
tools/mock_helpdesk.py
Mock help article database for the Customer Support Agent.

REAL_ARTICLES is the ground truth.
The hallucination scenario works by having the agent cite article IDs
that don't exist in REAL_ARTICLES — Phoenix flags the gap.
"""

REAL_ARTICLES = {
    "password-reset-101": {
        "id": "password-reset-101",
        "title": "How to reset your password",
        "url": "https://help.example.com/articles/password-reset-101",
        "content": (
            "To reset your password: 1) Click 'Forgot Password' on the login page. "
            "2) Enter your registered email address. 3) Click the reset link in the email "
            "within 10 minutes. 4) Choose a new password with at least 8 characters."
        ),
        "category": "account",
    },
    "2fa-setup-202": {
        "id": "2fa-setup-202",
        "title": "Setting up two-factor authentication",
        "url": "https://help.example.com/articles/2fa-setup-202",
        "content": (
            "Enable 2FA: 1) Go to Settings > Security. 2) Click 'Enable 2FA'. "
            "3) Scan the QR code with your authenticator app. 4) Enter the 6-digit code to confirm."
        ),
        "category": "security",
    },
    "billing-plans-301": {
        "id": "billing-plans-301",
        "title": "Understanding your billing plan",
        "url": "https://help.example.com/articles/billing-plans-301",
        "content": (
            "Plans: Starter ($49/mo), Pro ($99/mo, billed annually at $299/yr), "
            "Enterprise (custom). Invoices are sent on the 1st of each month. "
            "To upgrade: Settings > Billing > Change Plan."
        ),
        "category": "billing",
    },
    "data-export-401": {
        "id": "data-export-401",
        "title": "Exporting your data",
        "url": "https://help.example.com/articles/data-export-401",
        "content": (
            "Export options: Settings > Data Export. Formats: JSON (full export), "
            "CSV (table data only). Large exports may take up to 1 hour. "
            "You'll receive an email with a download link when ready."
        ),
        "category": "data",
    },
}

# These article IDs do NOT exist — used to trigger hallucination detection
NON_EXISTENT_ARTICLE_IDS = {
    "data-export-99999",
    "csv-format-guide-88888",
    "billing-refund-special-77777",
}


def search_help_articles(query: str, top_k: int = 3) -> dict:
    """
    Search help articles by keyword. Returns articles + confidence scores.
    Used by the Support Agent to retrieve relevant docs before answering.
    """
    q = query.lower()
    scored = []
    for article in REAL_ARTICLES.values():
        score = 0
        if any(word in article["content"].lower() for word in q.split()):
            score += 0.5
        if any(word in article["title"].lower() for word in q.split()):
            score += 0.4
        if article["category"] in q:
            score += 0.3
        if score > 0:
            scored.append({**article, "relevance_score": min(score, 1.0)})

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    top = scored[:top_k]
    top_confidence = top[0]["relevance_score"] if top else 0.0

    return {
        "results": top,
        "count": len(top),
        "top_confidence": top_confidence,
        "query": query,
    }


def get_article(article_id: str) -> dict:
    """Retrieve a specific article by ID. Returns error if not found."""
    if article_id in REAL_ARTICLES:
        return {"found": True, "article": REAL_ARTICLES[article_id]}
    return {"found": False, "error": f"Article '{article_id}' not found", "id": article_id}


def verify_urls(urls: list[str]) -> dict:
    """
    Check which cited URLs correspond to real help articles.
    Used by Phoenix evals to detect hallucinated citations.
    """
    real_urls = {a["url"] for a in REAL_ARTICLES.values()}
    results = {}
    for url in urls:
        results[url] = url in real_urls
    broken = [u for u, ok in results.items() if not ok]
    return {
        "checked": len(urls),
        "valid": len(urls) - len(broken),
        "broken": broken,
        "all_valid": len(broken) == 0,
    }
