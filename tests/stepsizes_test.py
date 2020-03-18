import numpy as np

from agents.agents import get_agent


def test_td_step_size():
    agent_info = {"step_size": 0.5, "representations": "TA", "num_states": 5}

    td = get_agent("TD")()
    td.agent_init(agent_info)

    assert td.step_size == agent_info.get("step_size")


def test_td_step_size_tile_coding():
    agent_info = {
        "step_size": 0.5,
        "representations": "TC",
        "tiles_per_dim": "10,10",
        "min_x": "0,0",
        "max_x": "1,1",
        "tilings": 5,
    }

    td = get_agent("TDTileCoding")()
    td.agent_init(agent_info)

    assert td.step_size == agent_info.get("step_size") / agent_info.get("tilings")


def test_etd_step_size_undiscounted():
    agent_info = {
        "step_size": 0.5,
        "representations": "TA",
        "num_states": 5,
        "discount_rate": 1.0,
        "trace_decay": 0.95,
        "interest": 1,
    }

    etd = get_agent("ETD")()
    etd.agent_init(agent_info)

    assert etd.step_size == agent_info.get("step_size")


def test_etd_step_size():
    agent_info = {
        "step_size": 0.5,
        "representations": "TA",
        "num_states": 5,
        "discount_rate": 0.25,
        "trace_decay": 0.95,
        "interest": 1,
    }

    etd = get_agent("ETD")()
    etd.agent_init(agent_info)

    assert etd.step_size == agent_info.get("step_size") / (
        (
            agent_info.get("interest")
            - agent_info.get("interest")
            * agent_info.get("trace_decay")
            * agent_info.get("discount_rate")
        )
        / (1 - agent_info.get("discount_rate"))
    )


def test_etd_step_size_tile_coding():
    agent_info = {
        "step_size": 0.5,
        "representations": "TC",
        "tiles_per_dim": "10,10",
        "min_x": "0,0",
        "max_x": "1,1",
        "tilings": 5,
        "discount_rate": 0.25,
        "trace_decay": 0.95,
        "interest": 1,
    }

    etd = get_agent("ETDTileCoding")()
    etd.agent_init(agent_info)

    M = (
        agent_info.get("interest")
        - agent_info.get("interest")
        * agent_info.get("trace_decay")
        * agent_info.get("discount_rate")
    ) / (1 - agent_info.get("discount_rate"))

    assert etd.step_size == agent_info.get("step_size") / agent_info.get("tilings") / M


def test_td_step_size_fourier():
    agent_info = {"step_size": 0.5, "representations": "F", "num_dims": 2, "order": 2}

    td = get_agent("TD")()
    td.agent_init(agent_info)

    C = td.FR.C
    num_features = td.FR.num_features

    step_sizes = np.full(td.FR.num_features, fill_value=agent_info.get("step_size"))
    for i in range(1, num_features):
        step_sizes[i] /= np.sqrt(np.sum(np.square(C[i])))

    assert np.array_equal(td.step_size, step_sizes)


def test_td_step_size_random_binary():
    agent_info = {
        "step_size": 0.5,
        "representations": "RB",
        "num_states": 5,
        "num_features": 3,
        "num_ones": 2,
        "seed": 0,
    }

    td = get_agent("TD")()
    td.agent_init(agent_info)

    num_ones = td.FR.num_ones

    td.step_size = agent_info.get("step_size") / num_ones


def test_etd_step_size_random_binary():
    agent_info = {
        "step_size": 0.5,
        "representations": "RB",
        "num_states": 5,
        "num_features": 3,
        "num_ones": 2,
        "seed": 0,
        "discount_rate": 0.25,
        "trace_decay": 0.95,
        "interest": 1,
    }

    etd = get_agent("ETD")()
    M = (
        agent_info.get("interest")
        - agent_info.get("interest")
        * agent_info.get("trace_decay")
        * agent_info.get("discount_rate")
    ) / (1 - agent_info.get("discount_rate"))
    etd.agent_init(agent_info)

    num_ones = etd.FR.num_ones

    etd.step_size = agent_info.get("step_size") / num_ones / M


def test_etd_step_size_fourier():
    agent_info = {
        "step_size": 0.5,
        "representations": "F",
        "num_dims": 2,
        "order": 2,
        "discount_rate": 0.25,
        "trace_decay": 0.95,
        "interest": 1,
    }

    etd = get_agent("ETD")()

    M = (
        agent_info.get("interest")
        - agent_info.get("interest")
        * agent_info.get("trace_decay")
        * agent_info.get("discount_rate")
    ) / (1 - agent_info.get("discount_rate"))

    etd.agent_init(agent_info)

    C = etd.FR.C
    num_features = etd.FR.num_features

    step_sizes = np.full(etd.FR.num_features, fill_value=agent_info.get("step_size"))
    for i in range(1, num_features):
        step_sizes[i] /= np.sqrt(np.sum(np.square(C[i])))

    step_sizes /= M

    assert np.array_equal(etd.step_size, step_sizes)
