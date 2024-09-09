PREFIX = """
You are working with a dataset in SAS. 
You should use the tools below to answer the question posed of you:"""

SUFFIX_NO_LIB = """
Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_LIB = """
This is the list of columns where libname=library:
{vcolumn}
 
If you need to modify data, create a new dataset with same name in work library and use that dataset for further analysis.
Begin!
Question: {input}
{agent_scratchpad}"""


PREFIX_FUNCTIONS = """
You are working with a dataset in SAS. """

FUNCTIONS_WITH_LIB = """
This is the list of columns where libname=library:
{vcolumn}

If you need to modify data, create a new dataset with same name in work library and use that dataset for further analysis.
Begin!
"""
