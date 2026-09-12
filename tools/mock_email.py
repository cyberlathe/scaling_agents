"""tools/mock_email.py — mock email sending."""
from datetime import datetime

_sent: list = []
_drafts: list = []


def send_email(to: str, subject: str, body: str, attachment: str = None) -> dict:
    email = {
        "id": f"MSG-{len(_sent)+1:04d}", "to": to, "subject": subject,
        "body_preview": body[:100] + "..." if len(body) > 100 else body,
        "attachment": attachment, "sent_at": datetime.now().isoformat(),
    }
    _sent.append(email)
    return {"success": True, "message_id": email["id"], "to": to,
            "warning": "Email delivered. Cannot be recalled."}


def draft_email(to: str, subject: str, body: str) -> dict:
    draft = {"id": f"DRAFT-{len(_drafts)+1:04d}", "to": to, "subject": subject,
             "body": body, "status": "draft"}
    _drafts.append(draft)
    return {"success": True, "draft_id": draft["id"], "status": "draft (not sent)"}


def get_sent() -> list:
    return _sent
