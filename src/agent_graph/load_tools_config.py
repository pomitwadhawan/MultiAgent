
import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()


class LoadToolsConfig:

    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]

        # Internet Search config
        self.tavily_search_max_results = int(
            app_config["tavily_search_api"]["tavily_search_max_results"])

      

        # NOShow RAG configs
        self.noshow_rag_llm = app_config["noshow_rag"]["llm"]
        self.noshow_rag_llm_temperature = float(
            app_config["noshow_rag"]["llm_temperature"])
        self.noshow_rag_embedding_model = app_config["noshow_rag"]["embedding_model"]
        self.noshow_rag_vectordb_directory = str(here(
            app_config["noshow_rag"]["vectordb"]))  # needs to be strin for summation in chromadb backend: self._settings.require("persist_directory") + "/chroma.sqlite3"
        self.noshow_rag_unstructured_docs_directory = str(here(
            app_config["noshow_rag"]["unstructured_docs"]))
        self.noshow_rag_k = app_config["noshow_rag"]["k"]
        self.noshow_rag_chunk_size = app_config["noshow_rag"]["chunk_size"]
        self.noshow_rag_chunk_overlap = app_config["noshow_rag"]["chunk_overlap"]
        self.noshow_rag_collection_name = app_config["noshow_rag"]["collection_name"]

        
         # `NO show` SQL Agent configs
        self.noshow_sqldb_directory  = str(here(
            app_config["noshow_sqlagent_configs"]["noshow_sqldb_dir"]))
        self.noshow_sqlagent_llm  = app_config["noshow_sqlagent_configs"]["llm"]
        self.noshow_sqlagent_llm_temperature  = float(
            app_config["noshow_sqlagent_configs"]["llm_temperature"])

        # Chinook SQL agent configs
        self.chinook_sqldb_directory = str(here(
            app_config["chinook_sqlagent_configs"]["chinook_sqldb_dir"]))
        self.chinook_sqlagent_llm = app_config["chinook_sqlagent_configs"]["llm"]
        self.chinook_sqlagent_llm_temperature = float(
            app_config["chinook_sqlagent_configs"]["llm_temperature"])

        # Graph configs
        self.thread_id = str(
            app_config["graph_configs"]["thread_id"])
