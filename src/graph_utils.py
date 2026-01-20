import os
import networkx as nx
import json
from typing import List, Dict, Set
from src.naming import get_canonical_name

def expand_selection_with_graph(selected_tables_real: List[str], schema_json_str: str, root_table_real: str | None = None) -> List[str]:
    """
    Analizza le tabelle selezionate dall'LLM e lo schema disponibile.
    Se le tabelle selezionate non sono connesse direttamente, trova e aggiunge
    le tabelle ponte necessarie usando il percorso minimo nel grafo delle FK.
    """

    DEBUG_GRAPH = os.getenv("GRAPH_DEBUG", "0") == "1"
    def _dbg(msg: str):
        if DEBUG_GRAPH:
            print(msg)

    # 1. Parsing dello schema recuperato dal Tool
    try:
        schema_list = json.loads(schema_json_str)
    except:
        return selected_tables_real # Fallback se il JSON è rotto

    if not schema_list:
        return selected_tables_real

    # 2. Costruzione del Grafo (Non orientato, perché i JOIN vanno in entrambe le direzioni)
    G = nx.Graph()
    
    # Mappe per convertire tra nomi Reali e Canonici
    # L'LLM lavora con i Reali, il Grafo con i Canonici (per sicurezza)
    canon_to_real = {}
    
    for table_data in schema_list:
        # Recuperiamo i nomi dal JSON di tools.py
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        
        if not canon_name:
            continue
        canon_to_real[canon_name] = real_name
        G.add_node(canon_name)
        
    schema_nodes = set(G.nodes)
    for table_data in schema_list:
        real_name = table_data.get("table") or table_data.get("table_name")
        canon_name = table_data.get("table_canonical") or get_canonical_name(real_name)
        if not canon_name:
            continue
        # Aggiungiamo gli archi basati sulle FK
        # Nota: tools.py dovrebbe passare le FK o dobbiamo inferirle. 
        # SE tools.py non passa esplicitamente le FK nel JSON finale, 
        # dobbiamo assicurarci che 'ingest_vector.py' le abbia salvate nei metadata 
        # e che 'tools.py' le includa nell'output.
        # Assumiamo che nel JSON ci sia un campo 'foreign_keys' o analizziamo 'original_ddl' se necessario.
        # PER ORA: Sfruttiamo il fatto che ingest_vector.py salva 'foreign_keys' nei metadata.
        # Bisogna assicurarsi che tools.py inoltri questo campo.
        
        # Se tools.py non passa 'foreign_keys' pulite, usiamo una logica di fallback o modifichiamo tools.py.
        # Qui assumiamo che tools.py restituisca i dati arricchiti.
        fks = table_data.get("foreign_keys", [])
        for fk in fks:
            # robustezza: accetta sia canonical che real, e canonicalizza comunque
            to_tbl = fk.get("to_table_canonical") or fk.get("to_table_real") or ""
            target_canon = get_canonical_name(str(to_tbl))
            if target_canon and target_canon in schema_nodes:
                G.add_edge(canon_name, target_canon)

    _dbg(f"[GRAPH] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")


    # 3) Normalizzazione deterministica: preserva l'ordine originale della lista LLM
    ordered_selected_canon = []
    seen = set()
    for name in selected_tables_real:
        c_name = get_canonical_name(name)
        if c_name in G.nodes and c_name not in seen:
            ordered_selected_canon.append(c_name)
            seen.add(c_name)

    if not ordered_selected_canon:
        return selected_tables_real

    # 4. Root esplicita: central_entity (se presente nel grafo), altrimenti fallback
    root_canon = get_canonical_name(root_table_real) if root_table_real else ""
    if root_canon and root_canon in G.nodes:
        # mettila come prima, senza duplicare
        ordered_selected_canon = [root_canon] + [x for x in ordered_selected_canon if x != root_canon]
    else:
        root_canon = ordered_selected_canon[0]

    _dbg(f"[GRAPH] root_real={root_table_real!r} root_canon={root_canon!r}")
    _dbg(f"[GRAPH] selected_canon_order={ordered_selected_canon}")

    if len(ordered_selected_canon) < 2:
        # già “connesso” o singola tabella
        return [canon_to_real.get(ordered_selected_canon[0], selected_tables_real[0])]
    
    # 5. Algoritmo di Riempimento (Steiner Tree approssimato tramite Shortest Path)
    # Prendiamo il primo nodo come 'root' e cerchiamo di collegare tutti gli altri a lui
    final_set_canon = set(ordered_selected_canon)
    root = ordered_selected_canon[0]

    for target in ordered_selected_canon[1:]:
        try:
            if nx.has_path(G, root, target):
                # Trova il percorso più breve: [root, bridge1, bridge2, target]
                path = nx.shortest_path(G, root, target)
                final_set_canon.update(path)
                _dbg(f"[GRAPH] path {root_canon} -> {target}: {path}")
            else:
                _dbg(f"[GRAPH] no-path {root_canon} -> {target}")
        except nx.NetworkXNoPath:
            _dbg(f"[GRAPH] NetworkXNoPath {root_canon} -> {target}")

    # 6) Output deterministico: prima le tabelle scelte dall'LLM (in ordine), poi i ponti (ordinati)
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
    
    _dbg(f"[GRAPH] final_real_names={final_real_names}")
    return final_real_names