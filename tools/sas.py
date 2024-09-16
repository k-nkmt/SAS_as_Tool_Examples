import re
from typing import  Annotated, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from langgraph.prebuilt import  InjectedState
import saspy
from IPython.display import display, HTML

# utility functions
def sanitize_input(query: str) -> str:
    """Sanitize input to the SAS.

    Remove whitespace, backtick & sas (if llm mistakes sas console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & sas from start
    query = re.sub(r"^(\s|`)*(?i:sas)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


def in_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# Tool for langchain(LCEL)  
class SASCode(BaseModel):
    """SAS Inputs."""
    query: str = Field(description="Should be a valid SAS command.")

class SimpleSASRepl(BaseTool):
    """Tool for running SAS code. Just returns listing."""

    name: str = "sas_repl"
    description: str = (
        "A SAS shell. Use this to execute SAS commands. Input should be a valid SAS command. "
    )
    args_schema: Type[BaseModel] = SASCode
    sanitize_input: bool = True
    sas:  Optional[saspy.SASsession] = Field(None)
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)

        if self.sas.SASpid is None:
            return "SAS not connected", ""
        
        ll = self.sas.submit(query, results="TEXT")
       # if self.sas.check_error_log:
        if re.search(r'\nERROR[ \d-]*:', ll["LOG"]):
            return "Error: "+self.sas.SYSERRORTEXT()+"\nCheck and fix your code."

        return ll["LST"]
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query)

# Tool to print SAS dataset, and example of using procedure.
class PrintInput(BaseModel):
    """Print Inputs."""
    dataset: str = Field(description="Dataset name to print.")

class PrintDS(BaseTool):
    """Tool for Get SAS dataset."""

    name: str = "print_ds"
    description: str = (
        "If specific variable names and conditions are not found in the Question, use this to get values of a SAS dataset. "
        "Input should be a valid SAS dataset name. "
    )
    args_schema: Type[BaseModel] = PrintInput
    sas: Optional[saspy.SASsession] = Field(None)
    obs: int = 5

    def _run(
        self,
        dataset: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        if self.sas.SASpid is None:
            return "SAS not connected."

        ll = self.sas.submit(f"proc print data = {dataset}(obs={self.obs}) noobs ; run ;")
        # if self.sas.check_error_log:
        if re.search(r'\nERROR[ \d-]*:', ll["LOG"]):
            return "Error: " + self.sas.SYSERRORTEXT()+"\nCheck dataset name."
        
        return ll["LST"]
    async def _arun(
        self,
        dataset: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, dataset)

# Tool for langchain(LCEL) with tool-calling model 
class SASInput(BaseModel):
    """SAS Inputs."""
    query: str = Field(description="Should be a valid SAS command.")
    direct: bool = Field(description="If True, the result is answered directly in HTML format as a FINAL ANSWER.")

class SASRepl(BaseTool):
    """Tool for running SAS code in a REPL."""

    name: str = "sas_repl"
    description: str = (
        "A SAS shell. Use this to execute SAS commands. Input should be a valid SAS command. "
        "By parameter direct, you can choose to return result directly. "
    )
    args_schema: Type[BaseModel] = SASInput
    response_format: str = "content_and_artifact"
    sanitize_input: bool = True
    display_html: bool = True
    sas:  Optional[saspy.SASsession] = Field(None)
    
    def _run(
        self,
        query: str,
        direct: bool,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[str, str]:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)

        if self.sas.SASpid is None:
            return "SAS not connected", ""

        ll = self.sas.submit(query, results= "HTML" if direct else "TEXT")

        # if self.sas.check_error_log:
        if re.search(r'\nERROR[ \d-]*:', ll["LOG"]):
            return "Error: " +self.sas.SYSERRORTEXT()+"\nCheck and fix your code.", ll["LOG"]
        if direct:
            self.return_direct = True
            if self.display_html and in_notebook():
                display(HTML(ll["LST"]))
            
            return "FINAL ANSWER.", ll["LST"]
        else:
            self.return_direct = False
            return ll["LST"], ""
    async def _arun(
        self,
        query: str,
        direct: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query, direct)


# tool for langgraph  
class SASInputState(SASInput):
    instructions: str = Field(description="Explanation of what you want to know from the results of queries to other assistants.")
    state:Annotated[dict, InjectedState] = Field(..., description="Agent state.")

class SASReplState(BaseTool):
    """Tool for running SAS code in a REPL with Agent State."""

    name: str = "sas_repl"
    description: str = (
        "A SAS shell. Use this to execute SAS commands. Input should be a valid SAS command. "
    )
    sanitize_input: bool = True
    display_html: bool = True
    args_schema: Optional[Type[BaseModel]] = SASInputState
    response_format: str = "content_and_artifact"
    
    def _run(
        self,
        query: str,
        direct: bool,
        instructions: str,
        state: dict,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[str, str]:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)

        if state["context"].sas.SASpid is None:
            return "SAS not connected.", ""
        
        ll = state["context"].sas.submit(query, results= "HTML" if direct else "TEXT")

        # if state["context"].sas.check_error_log:
        if re.search(r'\nERROR[ \d-]*:', ll["LOG"]):
            return "Error: "+state["context"].sas.SYSERRORTEXT()+"\nCheck and fix your code.",  ll["LOG"]
        if direct:
            if self.display_html and in_notebook():
                display(HTML(ll["LST"]))
            return "FINAL ANSWER.", ll["LST"]
        return "Got Listing Result." , ll["LST"]
    async def _arun(
        self,
        query: str,
        direct: bool,
        instructions: str,
        state: dict,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query, direct, instructions, state)