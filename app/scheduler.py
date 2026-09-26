"""最早可执行时刻推演（差分约束的最长路模型）。

每个提示点 ``p`` 满足：

* ``t[p] >= release[p]``（若提供 delay 覆盖，则 ``t[p] >= delay[p]``）；
* 对每条关系 ``(u -> v, min_gap=g)``，``t[v] >= t[u] + g``。

全部约束均为下界，因此逐点最早时刻等价于以"虚拟源点"（向每个点连权为
基础下界的边）为起点的最长路。最长路可用 Bellman-Ford 式的逐轮松弛求解，
无需枚举任何候选时间：

* 轮次与边的排列顺序无关，只影响收敛快慢，因此同一批规则无论怎样排列，
  得到的时间表完全相同；
* 若经过 ``n`` 轮完整松弛后仍可松弛，则存在源点可达的总权为正的有向环
  （正权环 => 时刻可沿环无限增长），返回 ``positive_cycle``；
* 否则结果即同时满足全部下界的最早时刻，再逐点检查 ``latest`` 上界。
"""

from __future__ import annotations

# 推演结果状态码（与 HTTP 响应中的稳定错误码保持一致）
STATUS_OK = "ok"
STATUS_POSITIVE_CYCLE = "positive_cycle"
STATUS_DEADLINE_EXCEEDED = "deadline_exceeded"

# 证据起点类型：某点的原始释放时间 / 本次 delay 覆盖值
KIND_RELEASE = "release"
KIND_DELAY = "delay"


class InconsistentScheduleError(Exception):
    """证据链与给定最早时刻对不上：实现缺陷或落库数据被篡改。

    API 层把它翻译成稳定的 ``result_inconsistent`` 错误；既有记录保持
    原样，绝不改写。
    """


def earliest_schedule(points, relations, delay=None):
    """求逐点最早可执行时刻。

    :param points: 已校验的提示点列表，元素为
        ``{"id": str, "release": int, "latest": Optional[int]}``。
    :param relations: 已校验的关系列表，元素为
        ``{"from": str, "to": str, "min_gap": int}``。
    :param delay: 可选的 ``{点 ID: 延误覆盖值}``，覆盖值不小于原 release。
    :returns:
        成功::

            {"status": "ok", "times": {id: int, ...}}  # 含全部提示点

        存在总间隔为正的有向环::

            {"status": "positive_cycle"}

        无正权环但有点最早时刻超过 latest::

            {"status": "deadline_exceeded",
             "violations": [{"id": str, "earliest": int, "latest": int}, ...],
             "times": {id: int, ...}}  # 收敛时刻，供违约点追溯证据链
    """
    delay = delay or {}
    ids = [p["id"] for p in points]
    n = len(ids)

    # 基础下界：虚拟源点 -> 点 p，权为 release（或 delay 覆盖）。
    times = {}
    for p in points:
        pid = p["id"]
        times[pid] = int(delay.get(pid, p["release"]))

    latest_by_id = {p["id"]: p.get("latest") for p in points}

    # 边以 (u, v, gap) 表示 t[v] >= t[u] + gap。
    edges = [(r["from"], r["to"], r["min_gap"]) for r in relations]

    # 至多 n 轮松弛即可得到无正权环图上的最长路。
    for _ in range(n):
        changed = False
        for u, v, gap in edges:
            candidate = times[u] + gap
            if candidate > times[v]:
                times[v] = candidate
                changed = True
        if not changed:
            break

    # 第 n+1 轮检测：仍能松弛 => 存在总权为正的有向环。
    for u, v, gap in edges:
        if times[u] + gap > times[v]:
            return {"status": STATUS_POSITIVE_CYCLE}

    # 无正权环：逐点核对 latest 上界（按 ID 升序，输出确定）。
    violations = []
    for pid in ids:
        latest = latest_by_id[pid]
        if latest is not None and times[pid] > latest:
            violations.append(
                {"id": pid, "earliest": times[pid], "latest": latest}
            )
    if violations:
        return {"status": STATUS_DEADLINE_EXCEEDED, "violations": violations,
                "times": times}

    return {"status": STATUS_OK, "times": times}


# ---- 因果解释：沿实际取紧的关系追溯证据链 ------------------------------------


def evidence_chains(points, relations, delay, times, targets):
    """在已收敛的 ``times`` 上为 ``targets`` 追溯因果证据链。

    每条证据从某点的基础下界（原始 release，或本次 delay 覆盖值）出发，沿
    **实际取紧**的关系（``times[to] == times[from] + min_gap``）到达目标点；
    逐边时差（取紧边上恰等于 min_gap）相加必然恰好等于 ``times[target]``。

    等价链（同样取紧、同样抵达目标点）按如下规则选出唯一结果：

    1. 关系数（经过的边数）最少者优先；
    2. 再按完整点 ID 序列的 UTF-8 字节序取最小者。

    零间隔环不会使追溯循环：绕环一周只增加关系数而不改变任何时刻，去掉环的
    链必然更优，因此最优链不含重复点，松弛必在有限轮内收敛。

    :param points: 已校验的提示点列表（同 :func:`earliest_schedule`）。
    :param relations: 已校验的关系列表。
    :param delay: 本次延误覆盖（可为 None）。
    :param times: 无正权环的收敛时刻表（:func:`earliest_schedule` 的输出；
        含 ``deadline_exceeded`` 结局中的时刻）。
    :param targets: 需要证据的点 ID 列表。
    :returns: ``{target: evidence}``，key 顺序与 ``targets`` 一致。
    :raises InconsistentScheduleError: 追溯结果与 ``times`` 对不上。
    """
    delay = delay or {}
    base = {}
    kinds = {}
    for p in points:
        pid = p["id"]
        if pid in delay:
            base[pid] = int(delay[pid])
            kinds[pid] = KIND_DELAY
        else:
            base[pid] = p["release"]
            kinds[pid] = KIND_RELEASE

    id_bytes = {pid: pid.encode("utf-8") for pid in base}

    # 取紧边：times[u] + min_gap == times[v]。
    tight_edges = [
        (r["from"], r["to"])
        for r in relations
        if times[r["from"]] + r["min_gap"] == times[r["to"]]
    ]

    # best[pid] = (关系数, 完整点 ID 字节序列, 点 ID 路径)；None = 尚未到达。
    # 起点：时刻恰好等于基础下界的点（证据从这里出发，0 条关系）。
    best = {}
    for pid in base:
        if times[pid] == base[pid]:
            best[pid] = (0, (id_bytes[pid],), (pid,))
        else:
            best[pid] = None

    # Bellman-Ford 式松弛：候选严格更优才替换。最优链必为简单链（至多
    # n-1 条边——含重复点的链可去掉零权环得到更优者），n 轮内必收敛；
    # 绕零间隔环的候选关系数更多，必然败北，追溯不会沿环打转。
    for _ in range(len(base)):
        changed = False
        for u, v in tight_edges:
            chain_u = best[u]
            if chain_u is None:
                continue
            key = (chain_u[0] + 1, chain_u[1] + (id_bytes[v],))
            current = best[v]
            if current is None or key < current[:2]:
                best[v] = (key[0], key[1], chain_u[2] + (v,))
                changed = True
        if not changed:
            break

    explanations = {}
    for target in targets:
        chain = best.get(target)
        if chain is None:
            raise InconsistentScheduleError(
                f"no tight evidence chain reaches point {target!r}")
        path = chain[2]
        steps = []
        total = base[path[0]]
        for u, v in zip(path, path[1:]):
            gap = times[v] - times[u]  # 取紧边的时差 == 该关系的 min_gap
            steps.append({"from": u, "to": v, "min_gap": gap})
            total += gap
        if total != times[target]:
            raise InconsistentScheduleError(
                f"evidence for {target!r} sums to {total}, "
                f"but earliest time is {times[target]}")
        explanations[target] = {
            "target": target,
            "start": {"id": path[0], "kind": kinds[path[0]],
                      "value": base[path[0]]},
            "steps": steps,
            "earliest": times[target],
        }
    return explanations
