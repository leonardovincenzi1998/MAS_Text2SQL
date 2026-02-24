import re
import os
import json
import torch
import networkx as nx
from typing import List, Optional
from safetensors.torch import save_file

# cleans and standardizes table or column names to a canonical format
# removes quotes, brackets, schema prefixes, and converts to lowercase
def get_canonical_name(name: str) -> str:
    if not name:
        return ""
    
    # remove extra spaces around dots
    s = name.strip()
    s = re.sub(r"\s*\.\s*", ".", s)

    # remove common outer quotes or brackets
    for wrap_l, wrap_r in [('"', '"'), ('`', '`'), ('[', ']'), ("'", "'")]:
        if s.startswith(wrap_l) and s.endswith(wrap_r):
            s = s[1:-1].strip()

    # keep only the last part if schema-qualified (e.g., dbo.TableName)
    if "." in s:
        s = s.split(".")[-1].strip()

    # final cleanup of any remaining structural characters
    s = s.strip().strip('"').strip("`").strip("[").strip("]").strip("'")
    
    return s.lower()


# analyzes selected tables and adds necessary bridge tables using shortest path in FK graph
def expand_selection_with_graph(
    selected_tables_real: List[str], 
    schema_json_str: str, 
    root_table_real: Optional[str] = None
) -> List[str]:
    DEBUG_GRAPH = os.getenv("GRAPH_DEBUG", "0") == "1"
    
    def _log_debug(msg: str) -> None:
        if DEBUG_GRAPH:
            print(msg)

    # parse the schema retrieved by the tool
    try:
        schema_list = json.loads(schema_json_str)
    except json.JSONDecodeError:
        return selected_tables_real

    if not schema_list:
        return selected_tables_real

    # build undirected graph because JOINs can go both ways
    G = nx.Graph()
    
    # maps to convert between real and canonical names for safety
    canon_to_real = {}
    
    for table_data in schema_list:
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        
        if not canon_name:
            continue
            
        canon_to_real[canon_name] = real_name
        G.add_node(canon_name)
        
    schema_nodes = set(G.nodes)
    
    # add edges based on foreign keys
    for table_data in schema_list:
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        
        if not canon_name:
            continue
        
        fks = table_data.get("foreign_keys", [])
        for fk in fks:
            # accept either canonical or real target, and canonicalize
            to_tbl = fk.get("to_table_canonical") or fk.get("to_table_real") or ""
            target_canon = get_canonical_name(str(to_tbl))
            
            if target_canon and target_canon in schema_nodes:
                G.add_edge(canon_name, target_canon)

    _log_debug(f"[GRAPH] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # deterministic normalization to preserve the original order of the LLM list
    ordered_selected_canon = []
    seen = set()
    for name in selected_tables_real:
        c_name = get_canonical_name(name)
        if c_name in G.nodes and c_name not in seen:
            ordered_selected_canon.append(c_name)
            seen.add(c_name)

    if not ordered_selected_canon:
        return selected_tables_real

    # explicitly place root central entity first if present
    root_canon = get_canonical_name(root_table_real) if root_table_real else ""
    if root_canon and root_canon in G.nodes:
        ordered_selected_canon = [root_canon] + [x for x in ordered_selected_canon if x != root_canon]
    else:
        root_canon = ordered_selected_canon[0]

    _log_debug(f"[GRAPH] root_real={root_table_real!r} root_canon={root_canon!r}")
    _log_debug(f"[GRAPH] selected_canon_order={ordered_selected_canon}")

    if len(ordered_selected_canon) < 2:
        return [canon_to_real.get(ordered_selected_canon[0], selected_tables_real[0])]
    
    # filling algorithm via shortest path
    final_set_canon = set(ordered_selected_canon)
    root = ordered_selected_canon[0]

    for target in ordered_selected_canon[1:]:
        try:
            if nx.has_path(G, root, target):
                path = nx.shortest_path(G, root, target)
                final_set_canon.update(path)
                _log_debug(f"[GRAPH] path {root_canon} -> {target}: {path}")
            else:
                _log_debug(f"[GRAPH] no-path {root_canon} -> {target}")
        except nx.NetworkXNoPath:
            _log_debug(f"[GRAPH] NetworkXNoPath {root_canon} -> {target}")

    # deterministic output: LLM tables first, then bridge tables
    final_real_names = []
    out_seen = set()

    for c_name in ordered_selected_canon:
        real = canon_to_real.get(c_name)
        if real and real not in out_seen:
            final_real_names.append(real)
            out_seen.add(real)

    for c_name in sorted(final_set_canon):
        real = canon_to_real.get(c_name)
        if real and real not in out_seen:
            final_real_names.append(real)
            out_seen.add(real)
    
    _log_debug(f"[GRAPH] final_real_names={final_real_names}")
    
    return final_real_names


if __name__ == "__main__":
    # path where the .bin was downloaded
    MODEL_DIR = "/scratch.hpc/leonardo.vincenzi/mas_text2sql/local_models/bge-m3"
    bin_file = os.path.join(MODEL_DIR, "pytorch_model.bin")
    safe_file = os.path.join(MODEL_DIR, "model.safetensors")

    print(f"🔄 Converto {bin_file} -> {safe_file}...")

    if not os.path.exists(bin_file):
        print("❌ Errore: pytorch_model.bin non trovato! Hai fatto il download?")
        exit(1)

    try:
        # load using pure torch to bypass transformers strict checks
        state_dict = torch.load(bin_file, map_location="cpu")
        
        # save in safe format
        save_file(state_dict, safe_file)
        print("✅ Conversione riuscita! Ora hai model.safetensors.")
        
        # optional: remove old file to save space and avoid confusion
        # os.remove(bin_file) 
    except Exception as e:
        print(f"❌ Errore durante la conversione: {e}")