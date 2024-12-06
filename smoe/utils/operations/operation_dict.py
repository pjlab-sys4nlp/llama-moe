def reverse_dict(input_dict, aggregate_same_results=True):
    output_dict = {}
    for key, value in input_dict.items():
        if value not in output_dict:
            output_dict[value] = key
        else:
            if aggregate_same_results:
                if not isinstance(output_dict[value], list):
                    output_dict[value] = [output_dict[value]]
                output_dict[value].append(key)
            else:
                raise ValueError(
                    "Input dictionary does not satisfy the one-to-one mapping condition."
                )
    return output_dict
