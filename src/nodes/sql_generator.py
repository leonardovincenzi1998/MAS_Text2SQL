import re
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models import AgentState
from src.database import DatabaseManager
from src.utils import validate_ast_and_format, prune_ddl_ast, format_table_metadata_as_sql_comment
from src.config import llm_sql
from src.prompts import SQL_GENERATOR_SYSTEM_PROMPT

# node 3: sql generation based on injected ddl, entities extraction and table selection

SQL_REGEX = r"```(?:sql|sqlite)?\s*(.*?)```"

async def run_sql_generator(state: AgentState) -> Dict[str, Any]:

    print("✍️  (SQL Generator) Iniezione DDL e generazione query...")

    # retrieve selected tables from previous agent
    selected_tables = state.get("selected_tables", [])
    selected_columns = state.get("selected_columns", {})

    if not selected_tables:
        return {"error": "No tables selected by Agent 2."}

    db_manager = DatabaseManager(state["db_path"])
    ddl_context = ""

    full_schema_list = state.get("parsed_schema", [])

    #1. clean DDL injection (Smart column pruning)
    # prune the DDL to include only relevant columns to save context window tokens and reduce hallucinations
    try:

        for table in selected_tables:

            raw_ddl = db_manager.get_table_ddl(table)

            tbl_data = next(
                (t for t in full_schema_list if t.get("table_name", t.get("table")) == table),
                None
            )

            # Find matching table metadata from the parsed schema list
            if tbl_data:

                colonne_pulite = set(tbl_data.get("columns", []))
                colonne_scelte_llm = state.get("selected_columns", {}).get(table, [])

                # Determine which columns to keep: use LLM selection if available, otherwise fallback to all valid columns
                allowed_cols = set(colonne_scelte_llm) if colonne_scelte_llm else colonne_pulite
                
                # Clean the AST of the DDL to physically remove unselected columns
                clean_ddl = prune_ddl_ast(raw_ddl, allowed_cols)

                # Generate SQL comments containing useful metadata (e.g., descriptions, foreign keys)
                meta_comment = format_table_metadata_as_sql_comment(
                    tbl_data,
                    allowed_cols
                )

                # Fallback to raw DDL if no metadata is found for the table
                ddl_context += f"-- Schema for {table}:\n{clean_ddl}\n{meta_comment}\n\n"

            else:

                ddl_context += f"-- Schema for {table}:\n{raw_ddl}\n\n"

        with open("debug_enriched_ddl.sql", "w", encoding="utf-8") as f:
            f.write(ddl_context)

        print("   💾 (SQL Generator) DDL arricchito salvato in 'debug_enriched_ddl.sql'")

    except Exception as e:

        return {"error": f"DDL extraction error: {str(e)}"}

    # 2. Format analytical context from Agent 1 (Intent, Entities, Filters)
    extraction = state.get("extraction_result")
    extracted_info = "None"

    if extraction:

        entities_str = "Nessuna"

        if hasattr(extraction, 'entities') and extraction.entities:

            entities_str = ", ".join(
                [f"[{e.category}: '{e.value}']" for e in extraction.entities]
            )

        extracted_info = (
            f"Intent: {extraction.intent}\n"
            f"Entities: {entities_str}\n"
            f"Filters: {extraction.filters}"
        )
    
    # Retrieve hints from the Vector DB to help the LLM match exact textual values present in the database
    entity_hints = state.get(
        "entity_hints",
        "Nessun hint disponibile sui valori testuali."
    )

    # 3. Prompt Construction and LLM Invocation
    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATOR_SYSTEM_PROMPT),
        ("human",
        """### CONTEXT

[ENRICHED DDL SCHEMA]
{ddl_context}

[EXTRACTED INFO]
{extracted_info}

[EXACT VALUE HINTS FROM VECTOR DB]
{entity_hints}

### USER QUESTION
{query}
""")
    ])

    chain = prompt | llm_sql

    try:

        response = await chain.ainvoke({
            "ddl_context": ddl_context,
            "extracted_info": extracted_info,
            "entity_hints": entity_hints,
            "query": state["user_query"]
        })

        # Parse LLM response content safely across different LangChain wrapper formats
        llm_output = response.content if hasattr(response, "content") else str(response)

        print("\n🧠 (SQL Generator) Reasoning LLM:\n")
        print(llm_output)

        # 4. SQL Parsing
        # Extract the SQL query from the markdown block using Regex
        match = re.search(SQL_REGEX, llm_output, re.DOTALL | re.IGNORECASE)

        if not match:

            return {
                "execution_status": False,
                "error_traceback": "SQL block not found in LLM output",
                "messages": [f"❌ Invalid SQL output:\n{llm_output}"]
            }

        clean_sql = match.group(1).strip()

        print("\n   🧾 (SQL Generator) SQL estratta:")
        print(clean_sql)

        print("   🔍 (AST Validator) Controllo conformità Tabelle e Colonne...")

        # 5. AST Validation
        # Parse the generated SQL's AST to ensure the LLM didn't hallucinate non-existent tables or columns
        final_sql, validation_error = validate_ast_and_format(
            clean_sql,
            selected_tables,
            selected_columns
        )

        # If AST validation fails, prepare the state with the error traceback for the Critic Agent to handle the retry
        if validation_error:

            print("   ⚠️ (AST Validator) Errore rilevato. Invio al Critic Agent.")

            return {
                "generated_sql": final_sql,
                "execution_status": False,
                "error_traceback": validation_error,
                "messages": [f"❌ Failed SQL Generation (AST):\n{validation_error}"]
            }

        print("   ✅ (AST Validator) Sintassi e Schema Linking confermati.")

        # Success: Return the valid SQL query to update the global state
        return {
            "generated_sql": final_sql,
            "messages": [f"✅ Generated SQL:\n{final_sql}"]
        }

    except Exception as e:

        return {"error": f"SQL Generator Error: {str(e)}"}