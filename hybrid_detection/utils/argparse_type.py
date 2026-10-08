

def float_or_none(str_in):
    return None if (str_in.lower() in {"none", "null"}) else float(str_in)
