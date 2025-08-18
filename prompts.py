DEFAULT_QUERIES = [
    ("Key risks last fiscal year", "Risk Factors"),
    ("Liquidity and capital resources", "MD&A"),
    ("Debt maturities, interest, covenants", "MD&A"),
    ("Capital allocation, dividends, buybacks, capex", "MD&A"),
]

SYSTEM_INSTRUCT = (
    "You are a cautious equity research assistant. Use ONLY the provided metrics and excerpts. "
    "Cite excerpts using [R#]. If unknown, say so. Be concise and neutral."
)

def build_prompt(tkr: str, kpi_json: dict, excerpts: list[dict]) -> str:
    """
    Compose a single prompt: header + compact KPIs + numbered excerpts with [R#] ids + tasks.
    Each excerpt dict: {text, meta:{file,page,section}}
    """
    lines = [
        SYSTEM_INSTRUCT,
        "",
        f"Company: {tkr}",
        "",
        "Metrics (deterministic JSON):",
        str(kpi_json),
        "",
        "Context excerpts:",
    ]
    for i, ex in enumerate(excerpts, 1):
        m = ex["meta"]
        cite = f"[R{i}] (Source: {m.get('file')} p.{m.get('page')} • {m.get('section')})"
        txt = ex["text"].replace("\n", " ")
        lines.append(f"{cite}\n{txt}\n")
    lines += [
        "Tasks:",
        "1) Operational performance (growth, margins, ROIC).",
        "2) Liquidity/solvency; call out refinancing/debt risks.",
        "3) Cash generation & capital allocation (dividends/buybacks/capex).",
        "4) 3–5 key risks, faithful to excerpts, each with [R#].",
        "5) One balanced watchlist paragraph (bull vs bear).",
        "Answer:",
    ]
    return "\n".join(lines)
