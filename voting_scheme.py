def vote(dict):
    for key in dict:
        dict[key] = (max(dict[key] ,key=dict[key].count))

    return dict
