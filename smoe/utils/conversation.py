from typing import List, Tuple, Union


class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    def __init__(self):
        # The name of this template
        self.name: str = "vicuna_v1.1"
        # The template of the system prompt
        self.system_template: str = "{system_message}"
        # The system message
        self.system_message: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        # The names of two roles
        self.roles: Tuple[str] = ("USER", "ASSISTANT")
        # All messages. Each item is (role, message).
        self.messages: List[List[str]] = []
        # The number of few shot examples
        self.offset: int = 0
        self.sep: str = " "
        self.sep2: str = "</s>"
        # Stop criteria (the default one is EOS token)
        self.stop_str: Union[str, List[str]] = None
        # Stops generation if meeting any token in this list
        self.stop_token_ids: List[int] = None

    def clear_msg(self):
        self.messages.clear()

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        seps = [self.sep, self.sep2]
        ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    @classmethod
    def parse(cls, instance: dict) -> str:
        conv = cls()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        sys_msg = instance.get("system_prompt")
        if sys_msg:
            conv.set_system_message(sys_msg)
        for j, turn in enumerate(instance["conversations"]):
            role = roles[turn["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, turn["value"])
        return conv.get_prompt()

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }
