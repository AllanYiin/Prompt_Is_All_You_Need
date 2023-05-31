import tiktoken


def estimate_used_tokens(string: str, model_name: str='gpt-3.5-turbo') -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding =tiktoken.encoding_for_model(model_name)
    except:
        encoding =tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens




def num_tokens_from_history(full_history, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for history_message in full_history:
          sub_tokens= 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in history_message.items():
              if key in ['role','name','content']:
                  sub_tokens += len(encoding.encode(value))
                  if key == "name":  # if there's a name, the role is omitted
                      sub_tokens += -1  # role is always required and always 1 token
      if 'total_tokens' not in history_message:
          history_message['total_tokens']=sub_tokens
      num_tokens+=sub_tokens
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")