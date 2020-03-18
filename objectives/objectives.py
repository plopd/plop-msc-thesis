from .square_error import RootSquareValueErrorObjective


def get_objective(name, true_values, mu, i):
    if name == "RMSVE":
        return RootSquareValueErrorObjective(true_values, mu, i)

    raise Exception("Unexpected objective given.")
