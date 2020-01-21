def get_action_from_policy(name, rand_generator=None):
    if name == "random-chain":
        left = 0
        right = 1
        return rand_generator.choice([left, right])

    raise Exception("Unexpected agent given")
