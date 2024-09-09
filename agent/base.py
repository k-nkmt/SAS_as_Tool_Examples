"""Agent for working with SAS datasets."""
# original: https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/agents/agent_toolkits/pandas  Licensed under the MIT License.

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from langchain.agents import (
    AgentType,
    create_openai_tools_agent,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    RunnableAgent,
    RunnableMultiActionAgent,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel, LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.tools import BaseTool

from .prompt import (
    FUNCTIONS_WITH_LIB,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_LIB,
    SUFFIX_WITH_LIB,
)

import saspy
from tools.sas import SimpleSASRepl, SASRepl, PrintDS

def _get_single_prompt(
    vcolumn: Union[None, saspy.SASdata],
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_vcolumn_in_prompt: Optional[bool] = True,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_vcolumn_in_prompt:
        suffix_to_use = SUFFIX_WITH_LIB
    else:
        suffix_to_use = SUFFIX_NO_LIB
    prefix = prefix if prefix is not None else PREFIX

    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)

    partial_prompt = prompt.partial()
    if "vcolumn" in partial_prompt.input_variables:
        partial_prompt = partial_prompt.partial(vcolumn=vcolumn)
    return partial_prompt

def _get_functions_single_prompt(
    vcolumn: Union[None, saspy.SASdata],
    *,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_vcolumn_in_prompt: Optional[bool] = True,
) -> ChatPromptTemplate:
    if include_vcolumn_in_prompt:
        suffix = (suffix or FUNCTIONS_WITH_LIB).format(vcolumn =vcolumn)
    prefix = prefix if prefix is not None else PREFIX_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def create_sas_dataset_agent(
    llm: LanguageModelLike,
    session_options: Optional[Dict[str, Any]] = {"results":"TEXT", "autoexec": "options nodate nonumber nocenter locale=en_us;"},
    lib_path:Union[str, List[str], None] = None,
    include_vcolumn_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    agent_type: Union[
        AgentType, Literal["openai-tools", "tool-calling","zero-shot-react-description"]
    ] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    allow_dangerous_code: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a SAS agent from an LLM and dataset(s).

    Security Notice:
        This agent relies on access to a SAS which can execute arbitrary code. 
        This can be dangerous and requires a specially sandboxed environment to be safely used. 
        Failure to run this code in a properly sandboxed environment can lead to arbitrary 
        code execution vulnerabilities, which can lead to data breaches, data loss, or other security incidents.

        Do not use this code with untrusted inputs, with elevated permissions,
        or without consulting your security team about proper sandboxing!

        You must opt-in to use this functionality by setting allow_dangerous_code=True.

    Args:
        llm: Language model to use for the agent. If agent_type is "tool-calling" then
            llm is expected to support tool calling.
        session_options: Options for sas session.
        lib_path: Path to the library to use for the agent. can set single path or a list of paths,
            assigned to the library name LIBRARY. 
        number_of_head_rows: Number of initial rows to include in prompt for tools.
        agent_type: One of "tool-calling", "openai-tools", "openai-functions", or
            "zero-shot-react-description". Defaults to "zero-shot-react-description".
            "tool-calling" is recommended over the legacy "openai-tools" and
            "openai-functions" types.
        callback_manager: DEPRECATED. Pass "callbacks" key into 'agent_executor_kwargs'
            instead to pass constructor callbacks to AgentExecutor.
        prefix: Prompt prefix string.
        suffix: Prompt suffix string.
        verbose: AgentExecutor verbosity.
        return_intermediate_steps: Passed to AgentExecutor init.
        max_iterations: Passed to AgentExecutor init.
        max_execution_time: Passed to AgentExecutor init.
        early_stopping_method: Passed to AgentExecutor init.
        agent_executor_kwargs: Arbitrary additional AgentExecutor args.
        extra_tools: Additional tools to give to agent on top of a PythonAstREPLTool.
        engine: One of "modin" or "pandas". Defaults to "pandas".
        allow_dangerous_code: bool, default False
            This agent relies on access to a SAS tool which can execute
            arbitrary code. This can be dangerous and requires a specially sandboxed
            environment to be safely used.
            Failure to properly sandbox this class can lead to arbitrary code execution
            vulnerabilities, which can lead to data breaches, data loss, or
            other security incidents.
            You must opt in to use this functionality by setting
            allow_dangerous_code=True.

        **kwargs: DEPRECATED. Not used, kept for backwards compatibility.

    Returns:
        An AgentExecutor with the specified agent_type agent and access to
        a SAS and any user-provided extra_tools.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            agent_executor = create_sas_agent(
                llm,
                agent_type="tool-calling",
                verbose=True
            )

    """
    if not allow_dangerous_code:
        raise ValueError(
            "This agent relies on access to a python repl tool which can execute "
            "arbitrary code. This can be dangerous and requires a specially sandboxed "
            "environment to be safely used. Please read the security notice in the "
            "doc-string of this function. You must opt-in to use this functionality "
            "by setting allow_dangerous_code=True."
            "For general security guidelines, please see: "
            "https://python.langchain.com/v0.2/docs/security/"
        )


    sas = saspy.SASsession(**(session_options ))

    if lib_path is not None:
        sas.saslib(libref="LIBRARY", path=lib_path, options="access=readonly")
    if include_vcolumn_in_prompt:
            df = sas.sasdata('vcolumn', 'sashelp', dsopts={"where":'libname="LIBRARY"', "keep":"LIBNAME MEMNAME NAME TYPE LABEL FORMAT", "rename":"(MEMNAME=dataset NAME=column)"}).to_df()
            vcolumn = str(df.to_markdown())
    else:
        vcolumn = ""
    
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        tools = [SimpleSASRepl(sas=sas)] + [PrintDS(sas=sas, obs=number_of_head_rows)]+ list(extra_tools)
        if include_vcolumn_in_prompt is not None and suffix is not None:
            raise ValueError(
                "If suffix is specified, include_vcolumn_in_prompt should not be."
            )
        prompt = _get_single_prompt(
            prefix=prefix,
            suffix=suffix,
            include_vcolumn_in_prompt=include_vcolumn_in_prompt,
            vcolumn=vcolumn,
        )
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
            runnable=create_react_agent(llm, tools, prompt),  # type: ignore
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
    elif agent_type in (AgentType.OPENAI_FUNCTIONS, "openai-tools", "tool-calling"):
        tools = [SASRepl(sas=sas)] + [PrintDS(sas=sas, obs=number_of_head_rows)]+ list(extra_tools)
        prompt = _get_functions_single_prompt(
            prefix=prefix,
            suffix=suffix,
            include_vcolumn_in_prompt=include_vcolumn_in_prompt,
            vcolumn=vcolumn,
        )
        if agent_type == AgentType.OPENAI_FUNCTIONS:
            runnable = create_openai_functions_agent(
                cast(BaseLanguageModel, llm), tools, prompt
            )
            agent = RunnableAgent(
                runnable=runnable,
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
        else:
            if agent_type == "openai-tools":
                runnable = create_openai_tools_agent(
                    cast(BaseLanguageModel, llm), tools, prompt
                )
            else:
                runnable = create_tool_calling_agent(
                    cast(BaseLanguageModel, llm), tools, prompt
                )
            agent = RunnableMultiActionAgent(
                runnable=runnable,
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
    else:
        raise ValueError(
            f"Agent type {agent_type} not supported at the moment. Must be one of "
            "'tool-calling', 'openai-tools', or 'zero-shot-react-description'."
        )
        
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )