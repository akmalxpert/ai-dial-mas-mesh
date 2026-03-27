from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class CalculationsAgentTool(BaseAgentTool):
    @property
    def deployment_name(self) -> str:
        return "calculations-agent"

    @property
    def name(self) -> str:
        return "calculations_agent"

    @property
    def description(self) -> str:
        return ("Calculations Agent. Primary goal is to work with calculations. "
                "Capable to make plotly graphics and chart bars. "
                "Equipped with: Python Code Interpreter (via MCP), and Simple calculator.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The request to the Calculations Agent"
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": "Whether to propagate the history of communication with this agent"
                }
            },
            "required": ["prompt"]
        }
