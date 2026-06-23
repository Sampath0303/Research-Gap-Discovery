import re


def clean_text(text: str) -> str:

    text = re.sub(
        r"\[[0-9]+\]",
        "",
        text
    )

    text = re.sub(
        r"Table\s+\d+.*",
        "",
        text
    )

    text = re.sub(
        r"Figure\s+\d+.*",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()