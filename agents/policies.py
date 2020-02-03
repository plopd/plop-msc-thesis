def get_action_from_policy(name, rand_generator):
    if name == "random-chain":
        left = 0
        right = 1
        return rand_generator.choice([left, right])
    elif name == "semi-random-puddle":
        north = 3
        east = 1
        return rand_generator.choice([north, east])

    raise Exception("Unexpected policy given.")
