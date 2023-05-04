import tiktoken


def estimate_used_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name='gpt-3.5-turbo')
    num_tokens = len(encoding.encode(string))
    return num_tokens