

def get_intersection_of_list(list_of_list_features):
    return list(set.intersection(*map(set, list_of_list_features)))
