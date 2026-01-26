import json


class StandardisedConversation:
    def __init__(self, root_data=None):
        self._next_id = 1  # root is defined as 0
        self._attempts = 0
        self.conversation = {
            0: {"children": [], "data": root_data}
        }

    def add_conversation(self, conversation_data: str):
        """Add an existing conversation from a JSON string."""
        loaded = json.loads(conversation_data)
        # Ensure all keys are int
        self.conversation = {int(k): v for k, v in loaded.items()}
        self._next_id = max(self.conversation.keys()) + 1

    def add_attempt(self):
        """Increment the attempt counter."""
        self._attempts += 1

# region root
    def get_root_id(self) -> int:
        """Get the root message ID."""
        return 0

    def get_root_data(self):
        """Get the root message."""
        return self.conversation[0]['data']

    def update_root_data(self, data):
        """Update the root message data."""
        self.conversation[0]['data'] = data
# endregion

# region messages
    def get_parent(self, message_id: int) -> int:
        """Get the parent ID of a given message."""
        message = self.conversation.get(message_id, None)
        if message:
            return message["parent"]
        return -1

    def add_message(self, parent_id: int, data) -> int:
        """Add a message to the conversation graph."""
        message_id = self._next_id
        self._next_id += 1

        if parent_id not in self.conversation and parent_id != 0:
            raise ValueError(f"Parent ID {parent_id} does not exist in the conversation.")

        message = {
            "parent": parent_id,
            "children": [],
            "data": data
        }
        self.conversation[message_id] = message

        if parent_id in self.conversation:
            self.conversation[parent_id]["children"].append(message_id)

        return message_id

    def get_message(self, message_id: int):
        """Retrieve a message by its ID."""
        return self.conversation.get(message_id, None)

    def get_message_data(self, message_id: int):
        """Retrieve the data of a message by its ID."""
        message = self.conversation.get(message_id, None)
        if message:
            return message["data"]
        return None
# endregion

# region utilities
    def get_message_total(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.conversation) - 1  # Exclude root

    def get_attempt_total(self) -> int:
        """Get the total number of attempts made in the conversation."""
        return self._attempts

    def get_path(self, message_id: int, root: bool = False) -> list:
        """Get the path from root to the specified message ID."""
        path = []
        current_id = message_id

        while current_id != 0:
            path.append(current_id)
            current_message = self.conversation.get(current_id, None)
            if current_message:
                current_id = current_message["parent"]
            else:
                break

        if root:
            path.append(0)

        path.reverse()
        return path

# endregion

    def __str__(self):
        return json.dumps(self.conversation)


class StandardisedMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return json.dumps(self.to_dict())
