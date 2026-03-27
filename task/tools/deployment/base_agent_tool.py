import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.tools.base_tool import BaseTool
from task.tools.models import ToolCallParams
from task.utils.stage import StageProcessor


class BaseAgentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Parse parameters
        arguments = json.loads(tool_call_params.tool_call.function.arguments)

        # 2. Create AsyncDial client and call the agent with streaming
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )

        messages = self._prepare_messages(tool_call_params)

        chunks = await client.chat.completions.create(
            messages=messages,
            stream=True,
            deployment_name=self.deployment_name,
            extra_headers={"x-conversation-id": tool_call_params.conversation_id}
        )

        # 3. Prepare variables
        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        # 4. Iterate through chunks
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content += delta.content
                    # Stream content to the stage for this tool call
                    if tool_call_params.stage:
                        tool_call_params.stage.append_content(delta.content)

                # Handle custom_content
                if hasattr(chunk.choices[0], 'custom_content') and chunk.choices[0].custom_content:
                    cc = chunk.choices[0].custom_content
                    if isinstance(cc, dict):
                        cc_dict = cc
                    elif hasattr(cc, 'model_dump'):
                        cc_dict = cc.model_dump(exclude_none=True)
                    else:
                        cc_dict = {}

                    # Set state
                    if cc_dict.get("state"):
                        custom_content.state = cc_dict["state"]

                    # Propagate attachments
                    if cc_dict.get("attachments"):
                        for att in cc_dict["attachments"]:
                            attachment = Attachment(**att) if isinstance(att, dict) else att
                            custom_content.attachments.append(attachment)
                            tool_call_params.choice.add_attachment(
                                **att if isinstance(att, dict) else att.model_dump(exclude_none=True))

                    # Stages propagation
                    if cc_dict.get("stages"):
                        for stage_data in cc_dict["stages"]:
                            stage_index = stage_data.get("index", 0)
                            if stage_index in stages_map:
                                # Propagate content to existing stage
                                propagated_stage = stages_map[stage_index]
                                if stage_data.get("name"):
                                    propagated_stage.append_name(stage_data["name"])
                                if stage_data.get("content"):
                                    propagated_stage.append_content(stage_data["content"])
                                if stage_data.get("status") == "completed":
                                    StageProcessor.close_stage_safely(propagated_stage)
                            else:
                                # Create new stage
                                stage_name = stage_data.get("name", f"Stage {stage_index}")
                                propagated_stage = StageProcessor.open_stage(
                                    choice=tool_call_params.choice,
                                    name=stage_name
                                )
                                stages_map[stage_index] = propagated_stage
                                if stage_data.get("content"):
                                    propagated_stage.append_content(stage_data["content"])
                                if stage_data.get("attachments"):
                                    for att in stage_data["attachments"]:
                                        propagated_stage.add_attachment(
                                            **att if isinstance(att, dict) else att.model_dump(exclude_none=True))
                                if stage_data.get("status") == "completed":
                                    StageProcessor.close_stage_safely(propagated_stage)

        # 5. Ensure stages are closed
        for stage in stages_map.values():
            StageProcessor.close_stage_safely(stage)

        # 6. Return Tool message
        return Message(
            role=Role.TOOL,
            content=StrictStr(content),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            name=StrictStr(tool_call_params.tool_call.function.name),
            custom_content=custom_content
        )

    def _prepare_messages(self, tool_call_params: ToolCallParams) -> list[dict[str, Any]]:
        # 1. Get params
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt", "")
        propagate_history = arguments.get("propagate_history", False)

        # 2. Prepare messages
        messages: list[dict[str, Any]] = []

        # 3. Collect proper history if propagate_history is True
        if propagate_history:
            all_messages = tool_call_params.messages
            for i, message in enumerate(all_messages):
                if message.role == Role.ASSISTANT:
                    if (message.custom_content
                            and message.custom_content.state
                            and isinstance(message.custom_content.state, dict)
                            and self.name in message.custom_content.state):
                        # Add the user message that is going before this assistant message
                        if i > 0 and all_messages[i - 1].role == Role.USER:
                            user_msg = all_messages[i - 1]
                            user_content = user_msg.content or ''
                            if user_msg.custom_content and user_msg.custom_content.attachments:
                                attachments_urls = '\n\nAttached files URLs:\n'
                                for att in user_msg.custom_content.attachments:
                                    if att.url:
                                        attachments_urls += f"{att.url}\n"
                                    elif att.reference_url:
                                        attachments_urls += f"{att.reference_url}\n"
                                user_content += attachments_urls
                            messages.append({
                                "role": Role.USER.value,
                                "content": user_content
                            })

                        # Add assistant message with refactored state
                        assistant_msg_copy = deepcopy(message)
                        agent_state = assistant_msg_copy.custom_content.state[self.name]
                        assistant_msg_copy.custom_content.state = agent_state
                        messages.append(assistant_msg_copy.model_dump(exclude_none=True))

        # 4. Add user message with prompt
        user_message: dict[str, Any] = {
            "role": Role.USER.value,
            "content": prompt
        }

        # Include custom_content (attachments) from the last user message
        if tool_call_params.messages:
            for msg in reversed(tool_call_params.messages):
                if isinstance(msg,
                              Message) and msg.role == Role.USER and msg.custom_content and msg.custom_content.attachments:
                    attachments_urls = '\n\nAttached files URLs:\n'
                    for att in msg.custom_content.attachments:
                        if att.url:
                            attachments_urls += f"{att.url}\n"
                        elif att.reference_url:
                            attachments_urls += f"{att.reference_url}\n"
                    user_message["content"] = prompt + attachments_urls
                    break

        messages.append(user_message)

        return messages
