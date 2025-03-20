from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class NoShowSQLAgentTool:
    """
    A tool for interacting with the No-Show SQL database using an LLM (Language Model) to generate and execute SQL queries.

    This tool enables users to ask no-show-related questions, which are transformed into SQL queries by a language model.
    The SQL queries are executed on the provided SQLite database, and the results are processed by the language model to
    generate a final answer for the user.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temperature: float) -> None:
        """
        Initializes the NoShowSQLAgentTool with the necessary configurations.

        Args:
            llm (str): The name of the language model to be used for generating and interpreting SQL queries.
            sqldb_directory (str): The directory path where the SQLite database is stored.
            llm_temperature (float): The temperature setting for the language model, controlling response randomness.
        """
        self.sql_agent_llm = ChatOpenAI(
            model=llm, temperature=llm_temperature)
        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
            Answer:
            """
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{sqldb_directory}")
        print(self.db.get_usable_table_names())  # Check available tables

        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(
            self.sql_agent_llm, self.db)
        answer_prompt = PromptTemplate.from_template(
            self.system_role)

        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        self.chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )


@tool
def query_noshow_sqldb(query: str) -> str:
    """Query the No-Show SQL Database and access all the related information. Input should be a search query."""
    agent = NoShowSQLAgentTool(
        llm=TOOLS_CFG.noshow_sqlagent_llm,
        sqldb_directory=TOOLS_CFG.noshow_sqldb_directory,
        llm_temperature=TOOLS_CFG.noshow_sqlagent_llm_temperature
    )
    response = agent.chain.invoke({"question": query})
    return response
