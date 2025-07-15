from pandasai_openai import OpenAI
import openai
import os
import pandasai as pai
from pandasai import Agent

LLM_CHAT_MODEL = "qwen-plus"


class PandasAILLMDashScope(OpenAI):
    """Custom OpenAI class for DashScope's Qwen models"""

    _supported_chat_models = [
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen3-235b-a22b",
        "qwen3-30b-a3b",
        "qwen3-32b",
        "qwen-turbo-2025-04-28",
        "qwen-plus-2025-04-28",
    ]

    def __init__(self, api_token: str, model: str = "qwen-plus", **kwargs):
        """
        Initialize the PandasAILLMDashScope class with DashScope's API base and Qwen model.

        Args:
            api_token (str): DashScope API key.
            model (str): Qwen model name (e.g., 'qwen-plus').
            **kwargs: Additional parameters for the OpenAI client.
        """
        # Set DashScope's API base - using the correct endpoint
        kwargs["api_base"] = kwargs.get(
            "api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Initialize the parent OpenAI class
        super().__init__(api_token=api_token, model=model, **kwargs)

        # Force chat model client for Qwen models
        self._is_chat_model = True
        self.client = (
            openai.OpenAI(**self._client_params).chat.completions
            if self.is_openai_v1()
            else openai.ChatCompletion
        )

    def is_openai_v1(self) -> bool:
        """
        Check if the openai library version is >= 1.0.

        Returns:
            bool: True if openai version is >= 1.0, False otherwise.
        """
        try:
            # For openai >= 1.0, the version is stored in openai.__version__
            version = openai.__version__
            major_version = int(version.split(".")[0])
            return major_version >= 1
        except AttributeError:
            # For older versions, assume pre-1.0
            return False


def setup_pandasai_llm():
    """Setup DashScope LLM for AI analysis"""
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print(
                "DASHSCOPE_API_KEY environment variable not set. Using mock analysis."
            )
            return None

        llm = PandasAILLMDashScope(api_token=api_key, model="qwen-plus")
        return llm
    except Exception as e:
        print(f"Failed to setup DashScope LLM: {e}")
        return None


def create_pandasai_agent(df, llm):
    try:
        # 创建自定义环境，避免IPython依赖
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np

        custom_env = {
            "pd": pd,
            "px": px,
            "go": go,
            "np": np,
            "DataFrame": pd.DataFrame,
            "Series": pd.Series,
        }

        pai.config.set(
            {
                "llm": llm,
                "verbose": True,
                "max_retries": 2,
                "enforce_privacy": True,
                "enable_logging": True,
                "enable_plotting": True,
                "save_charts": False,
                "plotting_engine": "plotly",
                "plotting_library": "plotly",
                "custom_whitelisted_dependencies": ["plotly", "pandas", "numpy"],
                "disable_plotting": False,
                "show_plot": False,  # Disable automatic plot display
                "custom_environment": custom_env,
                "code_execution_config": {
                    "last_message_is_code": True,
                    "work_dir": "./temp_analysis",
                    "use_docker": False,
                },
            }
        )
        agent = Agent([pai.DataFrame(df)])
        return agent
    except Exception as e:
        print(f"Failed to create pandasAI Agent: {e}")
        return None


def setup_chat_llm():
    """Setup Chat LLM for AI analysis"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not api_key or not api_base:
        print("Environment variables not set!")
        return None

    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        return client
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None
