import re

def get_canonical_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip()
    s = re.sub(r"\s*\.\s*", ".", s)

    # rimuovi quoting esterni comuni
    for wrap_l, wrap_r in [('"','"'), ('`','`'), ('[',']'), ("'","'")]:
        if s.startswith(wrap_l) and s.endswith(wrap_r):
            s = s[1:-1].strip()

    # schema-qualified -> ultimo pezzo
    if "." in s:
        s = s.split(".")[-1].strip()

    # pulizia finale
    s = s.strip().strip('"').strip("`").strip("[").strip("]").strip("'")
    return s.lower()
