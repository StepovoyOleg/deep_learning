from .models import RetrievedDocument, Source, WebResult


def report_source(doc: RetrievedDocument) -> Source:
    m = doc.metadata
    date = f", {m.report_date.strftime('%B %Y')}" if m.report_date else ""
    return Source(kind="report", citation=f"[Report: {m.report_name}{date}, p. {m.page}]")


def web_source(item: WebResult) -> Source:
    publisher = item.title.split(" - ")[-1] if " - " in item.title else item.title
    stamp = item.published_date.isoformat() if item.published_date else "date unavailable"
    return Source(kind="web", citation=f"[{publisher}, {stamp}, {item.url}]", url=item.url)
