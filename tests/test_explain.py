"""因果解释：证据链追溯、等价链裁决、零间隔环、截止违约、重算核对与重启读取。

证据契约：每条证据从某点的原始 release 或本次 delay 出发，沿实际取紧的
关系（t[to] == t[from] + min_gap）到达目标点；逐边时差相加恰好等于返回的
最早时刻。等价链先取关系数最少者，再按完整点 ID 序列的 UTF-8 字节序取
唯一结果；零间隔环不得使追溯循环。
"""

from __future__ import annotations

import json
import random
import sqlite3

from fastapi.testclient import TestClient

from app.main import create_app
from app.scheduler import (STATUS_DEADLINE_EXCEEDED, STATUS_OK,
                           earliest_schedule, evidence_chains)
from tests.conftest import make_template_payload


def pt(pid, release, latest=None):
    return {"id": pid, "release": release, "latest": latest}


def rel(u, v, gap):
    return {"from": u, "to": v, "min_gap": gap}


def explain_one(points, relations, delay, target):
    """调度器级快捷方式：推演成功并取 target 的证据。"""
    out = earliest_schedule(points, relations, delay)
    assert out["status"] == STATUS_OK
    return evidence_chains(points, relations, delay, out["times"],
                           [target])[target]


def assert_evidence_sound(evidence, expected_earliest):
    """证据通用不变量：起点值 + 逐边时差 == 返回的最早时刻。"""
    total = evidence["start"]["value"] + sum(
        step["min_gap"] for step in evidence["steps"])
    assert total == expected_earliest
    assert evidence["earliest"] == expected_earliest
    # 步骤首尾相接，且终点是目标点。
    path = [evidence["start"]["id"]]
    for step in evidence["steps"]:
        assert step["from"] == path[-1]
        path.append(step["to"])
    assert path[-1] == evidence["target"]
    # 链上无重复点（零间隔环不得使追溯循环）。
    assert len(path) == len(set(path))


# ---- 分支汇合 ---------------------------------------------------------------

def test_branch_reconvergence_follows_late_predecessor():
    # 菱形：a=0 -> b=5, c=20 -> d=max(1, 5+3, 20+0)=20；d 的证据必须沿
    # 真正取紧的晚到前驱 c，而不是录入顺序里先到的 b。
    points = [pt("a", 0), pt("b", 0), pt("c", 0), pt("d", 1)]
    relations = [rel("a", "b", 5), rel("a", "c", 20),
                 rel("b", "d", 3), rel("c", "d", 0)]
    ev = explain_one(points, relations, None, "d")
    assert ev["start"] == {"id": "a", "kind": "release", "value": 0}
    assert ev["steps"] == [{"from": "a", "to": "c", "min_gap": 20},
                           {"from": "c", "to": "d", "min_gap": 0}]
    assert_evidence_sound(ev, 20)


def test_reconvergence_after_delay_picks_delayed_chain():
    points = [pt("s", 0), pt("early", 0), pt("late", 0), pt("join", 0)]
    relations = [rel("s", "early", 5), rel("s", "late", 50),
                 rel("early", "join", 5), rel("late", "join", 50)]
    ev = explain_one(points, relations, None, "join")
    assert [s["from"] for s in ev["steps"]] == ["s", "late"]
    assert_evidence_sound(ev, 100)


# ---- 等价链裁决 -------------------------------------------------------------

def test_equivalent_chains_fewest_relations_wins():
    # t[d]=5 有两条取紧链：[a,d]（1 条关系）与 [b,c,d]（2 条关系）。
    points = [pt("a", 0), pt("b", 0), pt("c", 0), pt("d", 0)]
    relations = [rel("a", "d", 5), rel("b", "c", 2), rel("c", "d", 3)]
    ev = explain_one(points, relations, None, "d")
    assert ev["steps"] == [{"from": "a", "to": "d", "min_gap": 5}]
    assert ev["start"] == {"id": "a", "kind": "release", "value": 0}
    assert_evidence_sound(ev, 5)


def test_equivalent_chains_byte_order_of_full_id_sequence_wins():
    # 同为 1 条关系：[a,d] 与 [b,d]，按起点 ID 字节序取 a。
    points = [pt("a", 0), pt("b", 0), pt("d", 0)]
    relations = [rel("b", "d", 5), rel("a", "d", 5)]  # 录入顺序不影响裁决
    ev = explain_one(points, relations, None, "d")
    assert ev["start"]["id"] == "a"
    assert_evidence_sound(ev, 5)


def test_equivalent_chains_byte_order_compares_full_sequence():
    # 同为 2 条关系且首点相同：[a,z,d] 与 [a,y,d]，在第二点上 y < z。
    points = [pt("a", 0), pt("y", 0), pt("z", 0), pt("d", 0)]
    relations = [rel("a", "z", 1), rel("z", "d", 4),
                 rel("a", "y", 2), rel("y", "d", 3)]
    ev = explain_one(points, relations, None, "d")
    assert [ev["start"]["id"]] + [s["to"] for s in ev["steps"]] == ["a", "y", "d"]
    assert_evidence_sound(ev, 5)


def test_equivalent_chains_byte_order_uses_utf8_bytes():
    # 大写字母的 UTF-8 字节小于小写："Z"(0x5A) < "a"(0x61)。
    points = [pt("Z", 0), pt("a", 0), pt("d", 0)]
    relations = [rel("a", "d", 7), rel("Z", "d", 7)]
    ev = explain_one(points, relations, None, "d")
    assert ev["start"]["id"] == "Z"
    assert_evidence_sound(ev, 7)


# ---- 零间隔环 ---------------------------------------------------------------

def test_zero_gap_cycle_trace_terminates_with_simple_chain():
    # a->b->c->a 全 0 权：环上各点被拉平到 max(release)=3。追溯不得绕环。
    points = [pt("a", 1), pt("b", 2), pt("c", 3)]
    relations = [rel("a", "b", 0), rel("b", "c", 0), rel("c", "a", 0)]
    out = earliest_schedule(points, relations)
    assert out["status"] == STATUS_OK
    ev = evidence_chains(points, relations, None, out["times"],
                         ["a", "b", "c"])
    assert ev["c"]["steps"] == []  # c 的时刻就是自身 release
    # a 的证据：c --0--> a，恰 1 条关系，不绕环。
    assert ev["a"]["start"] == {"id": "c", "kind": "release", "value": 3}
    assert ev["a"]["steps"] == [{"from": "c", "to": "a", "min_gap": 0}]
    # b 的证据：c --0--> a --0--> b。
    assert [s["from"] for s in ev["b"]["steps"]] == ["c", "a"]
    for target in ("a", "b", "c"):
        assert_evidence_sound(ev[target], 3)


def test_zero_gap_self_loop_does_not_loop():
    points = [pt("a", 4)]
    ev = explain_one(points, [rel("a", "a", 0)], None, "a")
    assert ev["steps"] == []
    assert_evidence_sound(ev, 4)


def test_zero_cycle_with_external_feed_traces_to_feed():
    # 零权环 a<->b，外部 d --12--> a：a、b 的证据都落到 d 的 release。
    points = [pt("a", 0), pt("b", 0), pt("d", 0)]
    relations = [rel("a", "b", 0), rel("b", "a", 0), rel("d", "a", 12)]
    out = earliest_schedule(points, relations)
    ev = evidence_chains(points, relations, None, out["times"], ["a", "b"])
    assert ev["a"]["steps"] == [{"from": "d", "to": "a", "min_gap": 12}]
    assert [s["from"] for s in ev["b"]["steps"]] == ["d", "a"]
    for target in ("a", "b"):
        assert ev[target]["start"] == {"id": "d", "kind": "release",
                                       "value": 0}
        assert_evidence_sound(ev[target], 12)


# ---- delay 起点与随机DAG不变量 ----------------------------------------------

def test_evidence_starts_from_delay_override():
    points = [pt("a", 0), pt("b", 0), pt("c", 0)]
    relations = [rel("a", "b", 5), rel("b", "c", 6)]
    ev = explain_one(points, relations, {"a": 40}, "c")
    assert ev["start"] == {"id": "a", "kind": "delay", "value": 40}
    assert ev["steps"] == [{"from": "a", "to": "b", "min_gap": 5},
                           {"from": "b", "to": "c", "min_gap": 6}]
    assert_evidence_sound(ev, 51)


def test_target_at_own_release_has_empty_steps():
    points = [pt("a", 0), pt("solo", 9)]
    ev = explain_one(points, [rel("a", "a", 0)], None, "solo")
    assert ev["start"] == {"id": "solo", "kind": "release", "value": 9}
    assert ev["steps"] == []
    assert_evidence_sound(ev, 9)


def test_random_dags_every_point_evidence_sums_exactly():
    rng = random.Random(20260926)
    for trial in range(30):
        n = rng.randint(1, 12)
        ids = [f"p{i}" for i in range(n)]
        points = [pt(pid, rng.randint(0, 50)) for pid in ids]
        relations = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    relations.append(rel(ids[i], ids[j], rng.randint(0, 9)))
        delay = {pid: rng.randint(50, 80)
                 for pid in ids if rng.random() < 0.2}
        out = earliest_schedule(points, relations, delay)
        assert out["status"] == STATUS_OK
        evs = evidence_chains(points, relations, delay, out["times"], ids)
        for pid in ids:
            assert_evidence_sound(evs[pid], out["times"][pid])
            start = evs[pid]["start"]
            if start["kind"] == "delay":
                assert start["value"] == delay[start["id"]]
            else:
                assert start["value"] == points[ids.index(start["id"])]["release"]


# ---- 截止违约的证据（调度器级） ----------------------------------------------

def test_deadline_outcome_carries_times_for_violation_evidence():
    points = [pt("a", 0, latest=10), pt("b", 0, latest=5)]
    out = earliest_schedule(points, [rel("a", "b", 8)])
    assert out["status"] == STATUS_DEADLINE_EXCEEDED
    assert out["violations"] == [{"id": "b", "earliest": 8, "latest": 5}]
    # 违约点 b 的证据：从 a 的 release 出发，沿 a->b(8) 到达。
    ev = evidence_chains(points, [rel("a", "b", 8)], None, out["times"], ["b"])
    assert ev["b"]["start"] == {"id": "a", "kind": "release", "value": 0}
    assert ev["b"]["steps"] == [{"from": "a", "to": "b", "min_gap": 8}]
    assert_evidence_sound(ev["b"], 8)


# ---- API：推演时请求解释 ------------------------------------------------------

def _register(client, payload=None):
    resp = client.post("/templates", json=payload or make_template_payload())
    assert resp.status_code == 201, resp.text
    return resp.json()["template_id"]


def test_derivation_with_explain_returns_evidence(client):
    template_id = _register(client)
    resp = client.post("/derivations", json={
        "template_id": template_id,
        "delay": {"a": 30},
        "explain": ["d", "b"],
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["times"] == {"a": 30, "b": 35, "c": 50, "d": 50}
    assert list(body["explanations"].keys()) == ["b", "d"]  # 按 ID 升序
    ev_d = body["explanations"]["d"]
    assert ev_d == {
        "target": "d",
        "start": {"id": "a", "kind": "delay", "value": 30},
        "steps": [{"from": "a", "to": "c", "min_gap": 20},
                  {"from": "c", "to": "d", "min_gap": 0}],
        "earliest": 50,
    }
    ev_b = body["explanations"]["b"]
    assert ev_b["steps"] == [{"from": "a", "to": "b", "min_gap": 5}]
    assert ev_b["earliest"] == 35


def test_derivation_without_explain_response_unchanged(client):
    template_id = _register(client)
    resp = client.post("/derivations", json={"template_id": template_id})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # 旧响应逐项不变：不多一个键。
    assert set(body) == {"result_id", "template_id", "created_at", "delay",
                         "times", "points"}
    # explain 为空数组也等价于未请求。
    resp2 = client.post("/derivations", json={"template_id": template_id,
                                              "explain": []})
    assert resp2.status_code == 201
    assert set(resp2.json()) == set(body)


def test_derivation_explain_dedup_and_sort(client):
    template_id = _register(client)
    resp = client.post("/derivations", json={
        "template_id": template_id, "explain": ["d", "b", "d", "b"]})
    assert resp.status_code == 201, resp.text
    assert list(resp.json()["explanations"].keys()) == ["b", "d"]


def test_derivation_invalid_explain(client):
    template_id = _register(client)
    base = {"template_id": template_id}
    for bad in ({"explain": "d"}, {"explain": [1]}, {"explain": [""]},
                {"explain": ["ghost"]}, {"explain": ["a", "ghost"]}):
        resp = client.post("/derivations", json={**base, **bad})
        assert resp.status_code == 400, bad
        assert resp.json()["error"]["code"] == "invalid_explain", bad


def test_positive_cycle_with_explain_still_reports_original_error(client):
    payload = {
        "points": [{"id": "a", "release": 0}, {"id": "b", "release": 0}],
        "relations": [{"from": "a", "to": "b", "min_gap": 1},
                      {"from": "b", "to": "a", "min_gap": 1}],
    }
    template_id = _register(client, payload)
    resp = client.post("/derivations", json={"template_id": template_id,
                                             "explain": ["a"]})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "positive_cycle"
    assert "details" not in err  # 正权环没有时刻表，也没有证据


def test_deadline_exceeded_with_explain_gives_min_id_violator_evidence(
        client, db_path):
    payload = {
        "points": [
            {"id": "a", "release": 0, "latest": 100},
            {"id": "b", "release": 0, "latest": 5},
            {"id": "c", "release": 0, "latest": 2},
        ],
        "relations": [{"from": "a", "to": "b", "min_gap": 8},
                      {"from": "a", "to": "c", "min_gap": 9}],
    }
    template_id = _register(client, payload)
    resp = client.post("/derivations", json={"template_id": template_id,
                                             "explain": ["c"]})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "deadline_exceeded"
    assert err["details"]["violations"] == [
        {"id": "b", "earliest": 8, "latest": 5},
        {"id": "c", "earliest": 9, "latest": 2},
    ]
    # 只给 ID 最小的违约点 b 的证据（即使请求的是 c）。
    assert list(err["details"]["explanations"].keys()) == ["b"]
    ev = err["details"]["explanations"]["b"]
    assert ev["start"] == {"id": "a", "kind": "release", "value": 0}
    assert ev["steps"] == [{"from": "a", "to": "b", "min_gap": 8}]
    assert ev["earliest"] == 8

    # 失败推演不保存结果。
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_deadline_exceeded_without_explain_response_unchanged(client):
    payload = {
        "points": [{"id": "a", "release": 0, "latest": 10},
                   {"id": "b", "release": 0, "latest": 5}],
        "relations": [{"from": "a", "to": "b", "min_gap": 8}],
    }
    template_id = _register(client, payload)
    resp = client.post("/derivations", json={"template_id": template_id})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "deadline_exceeded"
    # 未请求解释：错误结构与旧契约逐项一致。
    assert err["details"] == {"violations": [
        {"id": "b", "earliest": 8, "latest": 5}]}


# ---- API：按结果 ID 读取同一解释 ----------------------------------------------

def test_get_result_with_explain_matches_post_explanation(client):
    template_id = _register(client)
    created = client.post("/derivations", json={
        "template_id": template_id,
        "delay": {"a": 30},
        "explain": ["b", "d"],
    }).json()
    result_id = created["result_id"]

    resp = client.get(f"/results/{result_id}", params=[("explain", "d"),
                                                       ("explain", "b")])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["explanations"] == created["explanations"]
    # 其余字段与未请求解释时逐项一致。
    plain = client.get(f"/results/{result_id}").json()
    assert "explanations" not in plain
    for key in ("result_id", "template_id", "created_at", "delay", "times",
                "points"):
        assert body[key] == plain[key]


def test_get_result_explain_after_restart(client, db_path):
    template_id = _register(client)
    created = client.post("/derivations", json={
        "template_id": template_id,
        "delay": {"a": 10},
        "explain": ["d"],
    }).json()
    result_id = created["result_id"]
    assert created["explanations"]["d"]["earliest"] == 30

    # 模拟容器重启：同一数据库文件重建应用，读到的解释逐字一致。
    restarted_app = create_app(db_path=db_path)
    with TestClient(restarted_app) as restarted:
        resp = restarted.get(f"/results/{result_id}?explain=d")
    assert resp.status_code == 200, resp.text
    assert resp.json()["explanations"] == created["explanations"]


def test_get_result_invalid_explain(client):
    template_id = _register(client)
    result_id = client.post(
        "/derivations", json={"template_id": template_id}).json()["result_id"]
    for query in ({"explain": "ghost"}, [("explain", "")],
                  [("explain", "a"), ("explain", "ghost")]):
        resp = client.get(f"/results/{result_id}", params=query)
        assert resp.status_code == 400, query
        assert resp.json()["error"]["code"] == "invalid_explain", query


def test_get_missing_result_with_explain_still_404(client):
    resp = client.get("/results/nonexistent-id?explain=a")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "result_not_found"


# ---- 重算核对：不一致时明确报错、不改写记录 ------------------------------------

def _corrupt_stored_times(db_path, result_id, new_times):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE results SET times_json = ? WHERE id = ?",
                     (json.dumps(new_times, sort_keys=True), result_id))
        conn.commit()
    finally:
        conn.close()


def _stored_times_json(db_path, result_id):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT times_json FROM results WHERE id = ?",
                            (result_id,)).fetchone()[0]
    finally:
        conn.close()


def test_inconsistent_result_rejected_and_record_not_rewritten(client, db_path):
    template_id = _register(client)
    created = client.post("/derivations", json={
        "template_id": template_id, "explain": ["d"]}).json()
    result_id = created["result_id"]

    # 直接篡改落库时刻（模拟数据损坏），再请求解释。
    _corrupt_stored_times(db_path, result_id,
                          {"a": 0, "b": 5, "c": 20, "d": 999})
    corrupted = _stored_times_json(db_path, result_id)

    resp = client.get(f"/results/{result_id}?explain=d")
    assert resp.status_code == 500
    err = resp.json()["error"]
    assert err["code"] == "result_inconsistent"
    assert "message" in err and err["message"]

    # 记录不被改写：损坏内容原样保留，既未修复也未删除。
    assert _stored_times_json(db_path, result_id) == corrupted

    # 未请求解释的旧读取路径不受核对影响，行为逐项不变。
    plain = client.get(f"/results/{result_id}")
    assert plain.status_code == 200
    assert plain.json()["times"]["d"] == 999
