#!/usr/bin/env python

from __future__ import print_function
import pybullet as p
from pddlstream.algorithms.meta import solve, create_parser
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test, get_tool_link
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z,get_aabb, \
    BLOCK_URDF, SMALL_BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system, create_box, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, YELLOW, CYAN, MAGENTA
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test
from pddlstream.language.constants import print_solution, PDDLProblem
from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn, get_cfree_obj_approach_pose_test
import itertools
import re
from pathlib import Path
import json
import os
import random
from typing import Dict, Any, Iterable, Optional, Tuple

def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed

def get_free_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1 = find_unique(lambda f: f[0] == 'freemotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        free_motion_fn = get_free_motion_gen(robot, obstacles, teleport)
        return free_motion_fn(q0, q1)
    return fn

def get_holding_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1, o, g = find_unique(lambda f: f[0] == 'holdingmotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        holding_motion_fn = get_holding_motion_gen(robot, obstacles, teleport)
        return holding_motion_fn(q0, q1, o, g)
    return fn

#######################################################
def _parse_cooked_targets_from_pddl(path):
    try:
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # (:goal  (and  (cooked foo) (cooked bar) ... ))
        m = re.search(r'\(:goal\s*(\(.+?\))\s*\)\s*', txt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        goal_sexpr = m.group(1)
        names = re.findall(r'\(cooked\s+([A-Za-z0-9_\-]+)\)', goal_sexpr, flags=re.IGNORECASE)
        return [n.lower() for n in names]
    except Exception as e:
        print(f"[goal-from-pddl][warn] failed to parse: {e}")
        return []

def pddlstream_from_problem(robot, movable=[], teleport=False, grasp_name='top', num_goal=3):
    #assert (not are_colliding(tree, kin_cache))

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, get_configuration(robot))
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose)]
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    for body in fixed:
        name = get_body_name(body)
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    # random goal selection
    # import random
    # primary_targets = movable[:6]
    # k = max(0, min(num_goal, len(primary_targets)))
    # chosen_targets = random.sample(primary_targets, k=k) if k > 0 else []
    # goal_terms = [('AtConf', conf)] + [('Cooked', b) for b in chosen_targets]
    # goal = tuple(['and'] + goal_terms) if len(goal_terms) > 1 else goal_terms[0]
    # print("goal:", goal)

    # TODO: goal을 랜덤하게 고르지 말고, problem pddl에서 가져와서 추가한다
    goal = None
    pddl_problem_path = f"/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem/kitchen{num_goal}_1.pddl"
    food_order = ['celery', 'radish', 'bacon', 'egg', 'chicken', 'apple']
    name_to_idx = {n: i for i, n in enumerate(food_order)}

    cooked_names = _parse_cooked_targets_from_pddl(pddl_problem_path)
    target_bodies = []
    if cooked_names:
        for nm in cooked_names:
            if nm in name_to_idx:
                idx = name_to_idx[nm]
                if 0 <= idx < len(movable):
                    target_bodies.append(movable[idx])
                else:
                    print(f"[goal-from-pddl][warn] '{nm}' index={idx} but movable has len={len(movable)}. skipped.")
            else:
                print(f"[goal-from-pddl][warn] unknown cooked target '{nm}'. skipped.")
        seen = set()
        target_bodies = [b for b in target_bodies if (b not in seen and not seen.add(b))]
    goal_terms = [('AtConf', conf)] + [('Cooked', b) for b in target_bodies]
    goal = tuple(['and'] + goal_terms) if len(goal_terms) > 1 else goal_terms[0]
    print(f"[goal-from-pddl] loaded {len(target_bodies)} target(s) from {pddl_problem_path}: {cooked_names}")

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################

def place(body, floor, x=0.0, y=0.0):
    set_pose(body, Pose(Point(x=x, y=y, z=stable_z(body, floor))))

def _pose_to_tuple(pose_obj: Dict[str, Any]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """{"position":[x,y,z], "quaternion":[qx,qy,qz,qw]} -> ((x,y,z),(qx,qy,qz,qw))"""
    pos = tuple(float(v) for v in pose_obj.get("position", [0,0,0]))
    quat = tuple(float(v) for v in pose_obj.get("quaternion", [0,0,0,1]))
    return (pos, quat)

def _find_entry(meta, prob_num, prob_idx, trial):
    for e in meta:
        if e.get("num") == prob_num and e.get("index") == prob_idx and e.get("trial") == trial:
            return e
    return None

def load_world(num_object=2, num_distractor=0, *, seed=None, food_perm=None,
               used_json_poses: bool = False,
               meta_json: Optional[str] = None,
               prob_num: Optional[int] = None,
               prob_idx: Optional[int] = None,
               trial: Optional[int] = None,):
    if seed is not None:
        random.seed(seed)

    set_default_camera()
    draw_global_system()
    with HideOutput():
        #add_data_path()
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/short_floor.urdf')
        sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5)))
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)))

        food_specs = [
            ('celery', GREEN),
            ('radish', BLUE),
            ('bacon', MAGENTA),
            ('egg', YELLOW),
            ('chicken', BROWN),
            ('apple', RED),
        ]

        foods = []
        for i in range(num_object):
            name, color = food_specs[i]
            # body = load_model(urdf, fixed_base=False)
            body = create_box(w=0.06, l=0.06, h=0.10, color=color)
            foods.append((name, body))

        distractors = []
        for i in range(num_distractor):
            # distractors.append(load_model(BLOCK_URDF, fixed_base=False))
            distractors.append(create_box(w=0.06, l=0.06, h=0.10, color=GREY))
        #cup = load_model('models/dinnerware/cup/cup_small.urdf',
        # Pose(Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot))
    # dump_body(robot)
    # wait_for_user()
    body_names = {
        sink: 'sink',
        stove: 'stove',
        floor: 'table',
    }
    for name, body in foods:
        body_names[body] = name
    for i, d in enumerate(distractors, start=1):
        body_names[d] = f'dis{i}'

    movable_bodies = [b for _, b in foods] + distractors

    base_food_positions = [
        (0.0, 0.5),
        (0.0, -0.5),
        (0.15, -0.5),
        (-0.15, 0.5),
        (-0.15, -0.5),
        (0.15, 0.5),
    ]

    if used_json_poses:
        wanted_names = [name for name, _ in foods]
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entry = _find_entry(meta, prob_num, prob_idx, trial)

        missing = [n for n in wanted_names if not entry["objects"].get(n) or "pose" not in entry["objects"].get(n)]
        if missing:
            raise ValueError(f"[used_json_poses=True] JSON에 pose가 없는 객체: {missing}")
        for name, body in foods:
            blk = entry["objects"].get(name)
            pos = tuple(blk["pose"]["position"])
            quat = tuple(blk["pose"]["quaternion"])
            set_pose(body, (pos, quat))
    else:
        if food_perm is None:
            indices = list(range(len(base_food_positions)))
            random.shuffle(indices)
        else:
            assert len(food_perm) == len(base_food_positions), \
                f"food_perm must be length {len(base_food_positions)}"
            indices = list(food_perm)

        for (name, body), idx in zip(foods, indices[:num_object]):
            x, y = base_food_positions[idx]
            place(body, floor, x=x, y=y,)

    grid = []
    xs = [-0.15, 0.0, 0.15] # fixed
    ys = [-0.4, 0.4, -0.6, 0.6]
    for yy in ys:
        for xx in xs:
            grid.append((xx, yy))
    for d, (xx, yy) in zip(distractors, grid):
        place(d, floor, x=xx, y=yy)

    return robot, body_names, movable_bodies, floor

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)

#######################################################
def _extract_goal_from_pddl(pddl_text: str) -> str:
    start = pddl_text.find("(:goal")
    if start == -1:
        return ""
    i = start
    depth = 0
    end = None
    while i < len(pddl_text):
        if pddl_text[i] == '(':
            depth += 1
        elif pddl_text[i] == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    return pddl_text[start:end] if end is not None else pddl_text[start:]

def _pose_to_json(pose_obj):
    pos, quat = pose_obj
    return {
        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
    }

def _aabb_extent(body_id):
    aabb = get_aabb(body_id)
    (xmin, ymin, zmin), (xmax, ymax, zmax) = aabb
    return [float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)]

def save_problems(
    *,
    domain_prefix="kitchen",
    pddl_dir="/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem",
    prob_num_range=(3,),
    prob_idx_range=(1,),
    n_trials=3,
    num_object=6,
    num_distractor=0,
    use_gui=False,
    seed=None
):
    out_dir = Path(pddl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = str(out_dir / "problems_meta_short.json")

    if seed is not None:
        random.seed(seed)

    base_indices = tuple(range(6))
    all_perms = list(itertools.permutations(base_indices, 6))
    random.shuffle(all_perms)

    def unique_perm_stream():
        for perm in all_perms:
            yield perm

    perm_iter = unique_perm_stream()
    entries = []

    for prob_num in prob_num_range:
        for prob_idx in prob_idx_range:
            pddl_path = Path(pddl_dir) / f"{domain_prefix}{prob_num}_{prob_idx}.pddl"
            if not pddl_path.exists():
                print(f"[save_problems] PDDL not found: {pddl_path}")
                goal_raw = ""
            else:
                pddl_text = pddl_path.read_text(encoding="utf-8", errors="ignore")
                goal_raw = _extract_goal_from_pddl(pddl_text).strip()

            for trial in range(1, n_trials + 1):
                try:
                    food_perm = next(perm_iter)
                except StopIteration:
                    raise RuntimeError("모든 순열을 소진했습니다. n_trials를 줄이거나 seed를 바꾸세요.")

                connect(use_gui=use_gui)
                try:
                    robot, body_names, movable_bodies, table_body = load_world(
                        num_object=num_object,
                        num_distractor=num_distractor,
                        seed=seed,
                        food_perm=food_perm,
                        used_json_poses=False,
                    )

                    food_names = {"celery","radish","bacon","egg","chicken","apple"}
                    objects_json = {}
                    for bid, name in body_names.items():
                        if name in food_names:
                            pose = get_pose(bid)
                            size = _aabb_extent(bid)
                            objects_json[name] = {
                                "bid": int(bid),
                                "size": size,
                                "pose": _pose_to_json(pose),
                            }

                    init_surfaces = {name: "table" for name in objects_json.keys()}

                    problem_name = f"{domain_prefix}{prob_num}_{prob_idx}"
                    entry = {
                        "problem_name": problem_name,
                        "pddl_path": str(pddl_path),
                        "num": int(prob_num),
                        "index": int(prob_idx),
                        "trial": int(trial),
                        "table_body": int(table_body),
                        "objects": objects_json,
                        "init_surfaces": init_surfaces,
                        "goal_raw": goal_raw,
                    }
                    entries.append(entry)

                finally:
                    disconnect()

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(entries)} entries to {out_json}")

def _as_list(x):
    if isinstance(x, (list, tuple)):
        return [_as_list(e) for e in x]
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)

def _joint_names(robot, joint_indices):
    names = []
    for j in joint_indices:
        try:
            names.append(p.getJointInfo(robot, j)[1].decode('utf-8'))
        except Exception:
            names.append(str(j))
    return names

def _stringify_action(act):
    """pddlstream plan 원소에서 (액션이름, 문자열 인자[]) 반환"""
    name = getattr(act, 'name', None)
    args = getattr(act, 'args', None)
    if name is None:
        if isinstance(act, (list, tuple)) and act and isinstance(act[0], str):
            name, args = act[0], act[1:]
        else:
            name, args = str(act), []
    return str(name), [str(a) for a in (args or [])]

def _serialize_conf_like(body, joints, values):
    return {
        "type": "conf",
        "body": int(body),
        "joints": _as_list(joints),
        "joint_names": _joint_names(body, joints),
        "values": _as_list(values),
    }

def _serialize_bodypath(bp):
    body = getattr(bp, 'body', -1)
    joints = getattr(bp, 'joints', [])
    path = getattr(bp, 'path', [])
    return {
        "type": "Trajectory",
        "path": [_serialize_conf_like(body, joints, q) for q in path],
    }

def _id2name_lookup(bid, id2name):
    try:
        b = int(bid)
    except Exception:
        return str(bid)
    return str(id2name.get(b, b))  # 못 찾으면 숫자를 문자열로

def _serialize_kitchen_cmd(obj, id2name=None):
    """
    id2name: {int body_id -> str name} 매핑. 없으면 숫자 그대로 출력.
    """
    cname = obj.__class__.__name__

    if cname == 'BodyPath':
        return _serialize_bodypath(obj)

    if cname in ('Attach', 'Detach'):
        out = {"type": cname}
        body_attr = getattr(obj, 'body', None)
        if body_attr is not None:
            try:
                bid = int(body_attr)
            except Exception:
                bid = body_attr

            if isinstance(bid, int) and id2name:
                out["body"] = str(id2name.get(bid, bid))
            else:
                out["body"] = str(bid)
        return out

    if hasattr(obj, 'body_paths'):
        return {"type": "Group", "commands": [_serialize_kitchen_cmd(c, id2name=id2name) for c in obj.body_paths]}

    if isinstance(obj, (list, tuple)):
        return {"type": "Group", "commands": [_serialize_kitchen_cmd(c, id2name=id2name) for c in obj]}

    return {"type": cname, "repr": repr(obj)}

def group_commands_by_action_kitchen(plan):
    """
    plan: pddlstream plan ([(name, args), ...] 형태)
    반환: [[cmd, cmd, ...], ...]  # 액션별 그룹
    """
    groups = []
    for step in plan:
        if isinstance(step, (list, tuple)) and len(step) >= 2:
            name, args = step[0], step[1]
        else:
            name = getattr(step, 'name', str(step))
            args = getattr(step, 'args', ())

        group = []
        if args:
            last = args[-1]
            lname = str(name).lower()
            if lname == 'place' and hasattr(last, 'reverse'):
                last = last.reverse()
            if hasattr(last, 'body_paths'):
                group = list(last.body_paths)
            elif hasattr(last, 'commands'):
                group = list(last.commands)
        groups.append(group)
    return groups

def export_plan_and_commands_grouped_kitchen(plan, grouped_commands, id2name):
    items = []
    for idx, (act, group) in enumerate(zip(plan, grouped_commands)):
        act_name, act_args = _stringify_action(act)
        items.append({
            "idx": idx,
            "pddl_action": act_name,
            "pddl_args": act_args,
            "commands": {"type": "Group", "commands": [_serialize_kitchen_cmd(c, id2name) for c in group]}
        })
    return {"plan_length": len(items), "items": items}



def main():
    parser = create_parser()
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    parser.add_argument('-t', '--max_time', default=600, type=int, help='The max time')
    parser.add_argument('--num_object', type=int, default=1,
                        help='Number of objects')
    parser.add_argument('--num_goal', type=int, default=1,
                        help='Number of primary target objects (from the first 6 movables) to include in the goal')
    parser.add_argument('--num_distractor', type=int, default=0,
                        help='Number of additional distractor blocks to spawn')
    parser.add_argument('--trial', type=int, default=1,
                        help='Number of additional distractor blocks to spawn')

    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=True)
    problem_json_path = "/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem/problems_meta_short.json"
    robot, names, movable, floor = load_world(num_object=args.num_object, num_distractor=args.num_distractor, used_json_poses=True, meta_json=problem_json_path, prob_num=args.num_goal, prob_idx=1, trial=args.trial)
    id2name = {int(bid): str(name) for bid, name in names.items()}
    print('Objects:', names)
    print("moveables:", movable)
    saver = WorldSaver()

    problem = pddlstream_from_problem(
        robot, movable=movable, teleport=args.teleport, num_goal=max(0, args.num_goal)
    )
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    # success_cost = 0 if args.optimal else INF
    # planner = 'ff-astar' if args.optimal else 'ff-wastar3'
    # search_sample_ratio = 2
    # max_planner_time = 10
    # # effort_weight = 0 if args.optimal else 1
    # effort_weight = 1e-3 if args.optimal else 1

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve(problem, algorithm=args.algorithm,
                             unit_costs=args.unit,
                             max_time=args.max_time, verbose=True, debug=False,
                             search_sample_ratio=1,
                             unit_efforts=True,)
            saver.restore()
    print_solution(solution)
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    command = postprocess_plan(plan)

    try:
        grouped = group_commands_by_action_kitchen(plan)
        export_dict = export_plan_and_commands_grouped_kitchen(plan, grouped, id2name)
        out_path = args.export_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, ensure_ascii=False, indent=2)
        print(f"[export] wrote {out_path}")
    except Exception as e:
        print(f"[export][warn] command json export failed: {e}")

    print("command:", command.body_paths)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        # wait_for_user('Execute?')
        #command.step()
        command.refine(num_steps=10).execute(time_step=0.001)
    # wait_for_user('Finish?')
    disconnect()

if __name__ == '__main__':
    # save_problems(
    #     domain_prefix="kitchen",
    #     pddl_dir="/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem",
    #     prob_num_range=range(3, 7),
    #     prob_idx_range=range(1, 2),
    #     n_trials=10,
    #     num_object=6,
    #     num_distractor=12,
    #     use_gui=True,
    #     seed=42,
    # )
    main()