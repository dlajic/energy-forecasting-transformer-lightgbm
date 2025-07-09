import os

from dotenv import load_dotenv

load_dotenv()  # einmalig beim Import


def use_dummy() -> bool:
    return os.getenv("USE_DUMMY_MODEL", "false").lower() == "true"
