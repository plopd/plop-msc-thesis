from .square_error import SquareValueErrorObjective


def get_objective(name, true_values, mu, i):
    if name == "MSVE":
        return SquareValueErrorObjective(true_values, mu, i)

    raise Exception("Unexpected objective given.")
