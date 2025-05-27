import io
from typing import Callable


def stream_reader(
    stream: io.TextIOWrapper | None,
    output_list: list[str],
    log_with_level: Callable,
) -> None:
    if stream:
        try:
            for line in iter(stream.readline, ""):
                line_content = line.strip()
                log_with_level(line_content)
                output_list.append(line_content)
        finally:
            stream.close()
