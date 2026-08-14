import heapq
import math


def _shortest_travel_times(model, origin):
    """
    Shortest t_min travel time from one physical origin to all physical nodes.
    """
    dist = [math.inf] * model.N
    dist[origin] = 0.0

    queue = [(0.0, origin)]

    while queue:
        current_dist, u = heapq.heappop(queue)

        if current_dist > dist[u]:
            continue

        for a in model.adj_out[u]:
            _, v_id = model.idx_to_arc_uv[a]
            v = model.node_id_to_idx[v_id]

            new_dist = current_dist + float(model.t_min_a[a])

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(queue, (new_dist, v))

    return dist


def compute_unavoidable_travel_baseline(domain, model):
    """
    Computing unavoidable baseline as shortest paths in infrastructure network.
    """
    baseline_raw = 0.0

    for o in range(model.N):

        # No need to run Dijkstra for origins without demand
        if model.D[o, :].sum() <= 0:
            continue

        dist = _shortest_travel_times(model, o)

        for d in range(model.N):
            demand = float(model.D[o, d])

            if demand <= 0:
                continue

            if not math.isfinite(dist[d]):
                raise ValueError(
                    f"No physical path between OD pair ({o}, {d}) "
                    f"with demand {demand}."
                )

            baseline_raw += demand * dist[d]

    time_weight = float(
        domain.config.get("travel_time_cost_mult", 1.0)
    )

    baseline_weighted = time_weight * baseline_raw

    return {
        "travel_baseline_raw": baseline_raw,
        "travel_baseline": baseline_weighted,
    }
