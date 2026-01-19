import networkx as nx
import json
from typing import List, Dict, Set
from src.naming import get_canonical_name

def expand_selection_with_graph(selected_tables_real: List[str], schema_json_str: str) -> List[str]:
    """
    Analizza le tabelle selezionate dall'LLM e lo schema disponibile.
    Se le tabelle selezionate non sono connesse direttamente, trova e aggiunge
    le tabelle ponte necessarie usando il percorso minimo nel grafo delle FK.
    """
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
        
        canon_to_real[canon_name] = real_name
        G.add_node(canon_name)
        
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
        fks = table_data.get("foreign_keys", []) # Assicurati che tools.py passi questo campo!
        if not fks:
             # Tentativo di recupero fallback se mancano le FK esplicite nell'output del tool
             pass 

        for fk in fks:
            target_canon = fk.get("to_table_canonical")
            if target_canon and target_canon in G.nodes:
                G.add_edge(canon_name, target_canon)

    # 3. Normalizzazione della selezione LLM
    selected_canon = set()
    for name in selected_tables_real:
        c_name = get_canonical_name(name)
        if c_name in G.nodes:
            selected_canon.add(c_name)
        else:
            # L'LLM ha allucinato una tabella non presente nel retrieval
            pass 

    if len(selected_canon) < 2:
        return selected_tables_real

    # 4. Algoritmo di Riempimento (Steiner Tree approssimato tramite Shortest Path)
    # Prendiamo il primo nodo come 'root' e cerchiamo di collegare tutti gli altri a lui
    final_set_canon = set(selected_canon)
    sorted_nodes = list(selected_canon)
    root = sorted_nodes[0]

    for target in sorted_nodes[1:]:
        try:
            if nx.has_path(G, root, target):
                # Trova il percorso più breve: [root, bridge1, bridge2, target]
                path = nx.shortest_path(G, root, target)
                final_set_canon.update(path)
        except nx.NetworkXNoPath:
            # I nodi sono in isole disconnesse del grafo
            pass

    # 5. Riconversione in Nomi Reali
    final_real_names = []
    for c_name in final_set_canon:
        if c_name in canon_to_real:
            final_real_names.append(canon_to_real[c_name])
            
    return final_real_names