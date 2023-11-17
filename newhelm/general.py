import uuid


def get_unique_id() -> str:
    return uuid.uuid4().hex
