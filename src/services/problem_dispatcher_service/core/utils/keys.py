DEFAULT_SEPARATOR: str = "#"
INPUT_KEY_SEPARATOR: str = "."


def join_key(*parts: str, separator: str = DEFAULT_SEPARATOR) -> str:
    cleaned = [p for p in parts if p]
    return separator.join(cleaned)


def split_key(key: str, separator: str = DEFAULT_SEPARATOR) -> list[str]:
    return key.split(separator)


def convert_key_separator(
    key: str,
    input_separator: str = INPUT_KEY_SEPARATOR,
    output_separator: str = DEFAULT_SEPARATOR,
) -> str:
    if input_separator == output_separator:
        return key
    return key.replace(input_separator, output_separator)
