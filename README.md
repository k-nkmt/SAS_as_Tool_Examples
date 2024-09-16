# SAS_as_Tool_Examples

[2024 SASユーザー総会](https://sas-user2024.ywstat.jp/) 資料  

LangChain,LangGraphによるエージェントと、その中でSASをツールとして呼び出す例です。  
そのまま使うものというより、適宜ツールを追加・調整したり、フローの構成などカスタムするベースとして参考にしもらえたらと思います。

エージェントは[pandas用](https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/agents/agent_toolkits/pandas)のものや、langgraphでは[リポジトリのExamples](https://github.com/langchain-ai/langgraph/tree/main/examples)や[チュートリアル](https://langchain-ai.github.io/langgraph/tutorials/)を参考にしています。

langchainを0.3系にアップデートしても動作することは確認しました。

2023年発表のリポジトリは[こちら](https://github.com/k-nkmt/SAS_API_LLM_Examples)

This repository contains the code used in the presentation at the [Japan SAS User Group Conference 2024](https://sas-user2024.ywstat.jp/). It provides examples of agents created with LangChain and LangGraph, and how to use SAS as a tool within these frameworks.  
examples.ipynb files contain explanations in Japanese, but the code comments and other annotations are written in English.