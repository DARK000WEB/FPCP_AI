from io import BytesIO
from pathlib import Path


def save_file(file_name: str, file_buffer: BytesIO) -> str:
    save_dir = Path("files")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name

    with open(file_path, "wb") as f:
        f.write(file_buffer.read())

    return str(file_path)
