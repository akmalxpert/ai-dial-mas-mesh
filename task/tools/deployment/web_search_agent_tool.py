from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class WebSearchAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "web-search-agent"

    @property
    def name(self) -> str:
        return "web_search_agent"

    @property
    def description(self) -> str:
        return ("WEB Search Agent. Performs research in WEB based on the user request. "
                "Equipped with: WEB search (DuckDuckGo via MCP) and is able to fetch WEB pages content.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The request to the WEB Search Agent"
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": "Whether to propagate the history of communication with this agent"
                }
            },
            "required": ["prompt"]
        }
