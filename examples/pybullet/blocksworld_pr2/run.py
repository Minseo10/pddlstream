#!/usr/bin/env python

from __future__ import print_function
import pybullet as p
from examples.pybullet.blocksworld_pr2.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    get_cfree_traj_grasp_pose_test, BASE_CONSTANT, distance_fn, move_cost_fn

from examples.pybullet.utils.pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, get_ik_ir_gen_only_arm, \
    get_stable_gen, get_grasp_gen, control_commands, get_table_gen
from examples.pybullet.utils.pybullet_tools.pr2_utils import get_arm_joints, ARM_NAMES, get_group_joints, \
    get_group_conf
from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, is_placement, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user
from examples.pybullet.namo.stream import get_custom_limits

from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.algorithms.common import SOLUTIONS
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from pddlstream.language.constants import Equal, And, print_solution, Exists, get_args, is_parameter, \
    get_parameter_name, PDDLProblem
from pddlstream.utils import read, INF, get_file_path, Profiler
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, DEBUG

from examples.pybullet.utils.pybullet_tools.pr2_primitives import apply_commands, State
from examples.pybullet.pr2.run import post_process
from examples.pybullet.utils.pybullet_tools.utils import draw_base_limits, WorldSaver, has_gui, str_from_object

from examples.pybullet.blocksworld_pr2.problems import PROBLEMS
import json
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Trajectory, GripperCommand, Attach, Detach, Pose as PoseCls, Conf as ConfCls, Commands as CommandsCls
)

# TODO: collapse similar streams into a single stream when reodering

def get_bodies_from_type(problem):
    bodies_from_type = {}
    for body, ty in problem.body_types:
        bodies_from_type.setdefault(ty, set()).add(body)
    return bodies_from_type

def pddlstream_from_problem(problem, init_surfaces, base_limits=None, collisions=True, teleport=False):
    robot = problem.robot
    table_id = problem.surfaces[0]

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {
        '@sink': 'sink',
        '@stove': 'stove',
    }

    #initial_bq = Pose(robot, get_pose(robot))
    initial_bq = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    init = [
        ('CanMove',),
        ('BConf', initial_bq),
        ('AtBConf', initial_bq),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
    ] + [('Sink', s) for s in problem.sinks] + \
           [('Stove', s) for s in problem.stoves] + \
           [('Connected', b, d) for b, d in problem.buttons] + \
           [('Button', b) for b, _ in problem.buttons]
    for arm in ARM_NAMES:
    #for arm in problem.arms:
        joints = get_arm_joints(robot, arm)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('Arm', arm), ('AConf', arm, conf), ('HandEmpty', arm), ('AtAConf', arm, conf)]
        if arm in problem.arms:
            init += [('Controllable', arm)]

    supported_blocks = set(init_surfaces.values()) & set(problem.movable)
    clear_blocks = set(problem.movable) - supported_blocks

    # block floating
    # table_pose = Pose(table_id, get_pose(table_id), init=True)
    # init += [('Pose', table_id, table_pose)]
    # init += [('AtPose', table_id, table_pose)]

    # init += [('Stackable', b1, b2) for b1 in problem.movable for b2 in problem.movable if b1 != b2]

    for body in problem.movable:
        pose = Pose(body, get_pose(body), init=True) # TODO: supported here
        init += [('Pose', body, pose), ('Graspable', body),
                 ('AtPose', body, pose), ('Stackable', body, None)]

        init += [('Stackable', body, table_id)] # block floating

        for surface in problem.surfaces:
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    for body, ty in problem.body_types:
        init += [('Type', body, ty)]

    # pddl initial state
    # on-table / stacked-on
    for body_id, surf_id in init_surfaces.items():
        if surf_id == table_id:
            init.append(('on-table', body_id))
        else:
            init.append(('stacked-on', body_id, surf_id))

    # clear
    occupied = set(init_surfaces.values()) - {table_id}
    for body_id in init_surfaces:
        if body_id not in occupied:
            init.append(('clear', body_id))

    bodies_from_type = get_bodies_from_type(problem)
    goal_literals = []
    if problem.goal_conf is not None:
        goal_conf = Conf(robot, get_group_joints(robot, 'base'), problem.goal_conf)
        init += [('BConf', goal_conf)]
        goal_literals += [('AtBConf', goal_conf)]
    for ty, s in problem.goal_on:
        bodies = bodies_from_type[get_parameter_name(ty)] if is_parameter(ty) else [ty]
        init += [('Stackable', b, s) for b in bodies]
        goal_literals += [('On', ty, s)]

        # pddl goal states
        if s == table_id:
            goal_literals += [('on-table', ty)]
        else:
            goal_literals += [('stacked-on', ty, s)]

    goal_literals += [('Holding', a, b) for a, b in problem.goal_holding] + \
                     [('Cleaned', b)  for b in problem.goal_cleaned] + \
                     [('Cooked', b)  for b in problem.goal_cooked]
    goal_formula = []
    for literal in goal_literals:
        parameters = [a for a in get_args(literal) if is_parameter(a)]
        if parameters:
            type_literals = [('Type', p, get_parameter_name(p)) for p in parameters]
            goal_formula.append(Exists(parameters, And(literal, *type_literals)))
        else:
            goal_formula.append(literal)
    goal_formula = And(*goal_formula)

    custom_limits = {}
    if base_limits is not None:
        custom_limits.update(get_custom_limits(robot, problem.base_limits))

    stream_map = {
        # 'sample-table-pose': from_gen_fn(get_table_gen(problem, collisions=collisions)),
        'sample-pose': from_gen_fn(get_stable_gen(problem, collisions=collisions)),
        'sample-grasp': from_list_fn(get_grasp_gen(problem, collisions=collisions)),
        #'sample-grasp': from_gen_fn(get_grasp_gen(problem, collisions=collisions)),
        'inverse-kinematics': from_gen_fn(get_ik_ir_gen_only_arm(problem, custom_limits=custom_limits,
                                                                collisions=collisions, teleport=teleport)),
        # 'inverse-kinematics': from_gen_fn(get_ik_ir_gen(problem, custom_limits=custom_limits,
        #                                                 collisions=collisions, teleport=teleport)),
        'plan-base-motion': from_fn(get_motion_gen(problem, custom_limits=custom_limits,
                                                   collisions=collisions, teleport=teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(robot, collisions=collisions)),
        #'test-cfree-traj-grasp-pose': from_test(get_cfree_traj_grasp_pose_test(problem, collisions=collisions)),

        #'MoveCost': move_cost_fn,
        'Distance': distance_fn,
    }
    #stream_map = DEBUG

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal_formula)

#######################################################
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

def _iter_flat(cmd):
    """Commands/list/tuple 안쪽까지 전부 원자 Command로 평탄화"""
    if cmd is None:
        return
    if isinstance(cmd, CommandsCls):
        for c in cmd.commands:
            yield from _iter_flat(c)
    elif isinstance(cmd, (list, tuple)):
        for c in cmd:
            yield from _iter_flat(c)
    else:
        yield cmd

def _flatten_all(commands):
    """post_process가 반환한 commands 컨테이너 전체를 평탄화"""
    flat = []
    if isinstance(commands, (list, tuple)):
        for c in commands:
            flat.extend(list(_iter_flat(c)))
    else:
        flat.extend(list(_iter_flat(commands)))
    return flat

def _stringify_action(act):
    name = getattr(act, 'name', None)
    args = getattr(act, 'args', None)
    if name is None:
        if isinstance(act, (list, tuple)) and act and isinstance(act[0], str):
            name, args = act[0], act[1:]
        else:
            name, args = str(act), []
    return str(name), [str(a) for a in (args or [])]

def group_commands_by_action(plan, raw_commands):
    """
    plan: pddlstream plan
    raw_commands: post_process()가 반환한 commands (보통 list)
    반환: [ [cmd, cmd, ...], [cmd, ...], ... ]  # 액션별 커맨드 그룹
    """
    flat = _flatten_all(raw_commands)
    it = iter(flat)
    out = []

    def take_until_traj():
        g = []
        while True:
            c = next(it)
            g.append(c)
            if isinstance(c, Trajectory):
                break
        return g

    def take_until_after(marker_cls):
        g = []
        seen_marker = False
        seen_traj_after = False
        while True:
            c = next(it)
            g.append(c)
            if isinstance(c, marker_cls):
                seen_marker = True
                continue
            if seen_marker and isinstance(c, Trajectory):
                seen_traj_after = True
            if seen_marker and seen_traj_after:
                break
        return g

    for act in plan:
        name, _ = _stringify_action(act)
        lname = name.lower()
        if 'move_base' in lname:
            out.append(take_until_traj())
        elif lname in ('unstack', 'pick'):
            out.append(take_until_after(Attach))
        elif lname in ('stack', 'place'):
            out.append(take_until_after(Detach))
        else:
            # 기본: Trajectory 하나까지 집어넣기
            out.append(take_until_traj())
    return out

# ==== 직렬화 ====

def _serialize_waypoint(wp):
    if isinstance(wp, PoseCls):
        pos, orn = wp.value
        return {"type": "base_pose", "pos": _as_list(pos), "orn": _as_list(orn)}
    if isinstance(wp, ConfCls):
        return {
            "type": "conf",
            "body": int(wp.body),
            "joints": _as_list(wp.joints),
            "joint_names": _joint_names(wp.body, wp.joints),
            "values": _as_list(wp.values),
        }
    return {"type": type(wp).__name__, "repr": repr(wp)}

def _serialize_command(cmd):
    if isinstance(cmd, Trajectory):
        return {"type": "Trajectory", "path": [_serialize_waypoint(w) for w in cmd.path]}
    if isinstance(cmd, GripperCommand):
        return {"type": "GripperCommand", "arm": cmd.arm, "left": (cmd.arm == "left"),
                "target": _as_list(cmd.position), "teleport": bool(cmd.teleport)}
    if isinstance(cmd, Attach):
        return {"type": "Attach", "arm": cmd.arm, "left": (cmd.arm == "left"), "body": int(cmd.body)}
    if isinstance(cmd, Detach):
        return {"type": "Detach", "arm": cmd.arm, "left": (cmd.arm == "left"), "body": int(cmd.body)}
    if isinstance(cmd, CommandsCls):
        return {"type": "Group", "commands": [_serialize_command(c) for c in cmd.commands]}
    if isinstance(cmd, (list, tuple)):
        return {"type": "Group", "commands": [_serialize_command(c) for c in cmd]}
    return {"type": type(cmd).__name__, "repr": repr(cmd)}

def export_plan_and_commands_grouped(plan, grouped_commands):
    items = []
    for idx, (act, group) in enumerate(zip(plan, grouped_commands)):
        act_name, act_args = _stringify_action(act)
        items.append({
            "idx": idx,
            "pddl_action": act_name,
            "pddl_args": act_args,
            "commands": {"type": "Group", "commands": [_serialize_command(c) for c in group]}
        })
    return {"plan_length": len(items), "items": items}

def main(verbose=True):
    # TODO: could work just on postprocessing
    # TODO: try the other reachability database
    # TODO: option to only consider costs during local optimization

    parser = create_parser()
    parser.add_argument('-problem', default='packed', help='The name of the problem to solve')
    parser.add_argument('-n', '--number', default=5, type=int, help='The number of objects')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions')
    parser.add_argument('-deterministic', action='store_true', help='Uses a deterministic sampler')
    parser.add_argument('-optimal', action='store_true', help='Runs in an anytime mode')
    parser.add_argument('-t', '--max_time', default=600, type=int, help='The max time')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    parser.add_argument('--problem_path',  # note the two dashes
                        type=str,
                        help='Path to problem PDDL file')
    args = parser.parse_args()
    print('Arguments:', args)

    problem_fn_from_name = {fn.__name__: fn for fn in PROBLEMS}
    if args.problem not in problem_fn_from_name:
        raise ValueError(args.problem)
    problem_fn = problem_fn_from_name[args.problem]

    connect(use_gui=True)
    with HideOutput():
        meta_json = "/home/minseo/robot_ws/src/tamp_llm/experiments/blocksworld_pr/problem/problems_meta.json"
        # problem, init_surfaces, blocks_by_color = problem_fn(problem_pddl_path=args.problem_path)
        problem, init_surfaces, blocks_by_color = problem_fn(meta_json=meta_json, prob_num=args.prob_num, prob_idx=args.prob_idx, trial=args.trial, arm='left', grasp_type='top', used_json_poses=True)
    draw_base_limits(problem.base_limits, color=(1, 0, 0))
    saver = WorldSaver()

    #handles = []
    #for link in get_group_joints(problem.robot, 'left_arm'):
    #    handles.append(draw_link_name(problem.robot, link))
    #wait_for_user()

    pddlstream_problem = pddlstream_from_problem(problem, init_surfaces, collisions=not args.cfree, teleport=args.teleport)
    stream_info = {
        'inverse-kinematics': StreamInfo(),
        'plan-base-motion': StreamInfo(overhead=1e1),

        'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
        'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
        'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=verbose), # TODO: apply to arm and base trajs
        #'test-cfree-traj-grasp-pose': StreamInfo(verbose=verbose),

        'Distance': FunctionInfo(p_success=0.99, opt_fn=lambda q1, q2: BASE_CONSTANT),
        #'MoveCost': FunctionInfo(lambda t: BASE_CONSTANT),
    }
    #stream_info = {}

    _, _, _, stream_map, init, goal = pddlstream_problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    success_cost = 0 if args.optimal else INF
    planner = 'ff-astar' if args.optimal else 'ff-wastar3'
    search_sample_ratio = 1 # low ratio(0.5): more serach, high ration(2): more sampling
    max_planner_time = 10
    # effort_weight = 0 if args.optimal else 1
    effort_weight = 1e-3 if args.optimal else 1

    with Profiler(field='tottime', num=25): # cumtime | tottime
        with LockRenderer(lock=not args.enable):
            solution = solve(pddlstream_problem, algorithm=args.algorithm, stream_info=stream_info,
                             planner=planner, max_planner_time=max_planner_time,
                             unit_costs=args.unit, success_cost=success_cost,
                             max_time=args.max_time, verbose=True, debug=False,
                             unit_efforts=True, effort_weight=effort_weight,
                             search_sample_ratio=search_sample_ratio)
            saver.restore()


    cost_over_time = [(s.cost, s.time) for s in SOLUTIONS]
    for i, (cost, runtime) in enumerate(cost_over_time):
        print('Plan: {} | Cost: {:.3f} | Time: {:.3f}'.format(i, cost, runtime))
    #print(SOLUTIONS)
    print_solution(solution)
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    with LockRenderer(lock=not args.enable):
        commands = post_process(problem, plan, teleport=args.teleport)
        saver.restore()

    try:
        grouped = group_commands_by_action(plan, commands)
    except StopIteration:
        flat = _flatten_all(commands)
        grouped = [[flat[i]] if i < len(flat) else [] for i in range(len(plan))]
        print("[export][warn] grouping ran out of commands; fell back to naive pairing")

    export_dict = export_plan_and_commands_grouped(plan, grouped)
    with open(args.export_json, 'w') as f:
        json.dump(export_dict, f, indent=2)
    print(f"[export] wrote {args.export_json}")

    draw_base_limits(problem.base_limits, color=(1, 0, 0))
    # wait_for_user()
    if args.simulate:
        control_commands(commands)
    else:
        time_step = None if args.teleport else 0.01
        apply_commands(State(), commands, time_step)
    # wait_for_user()
    disconnect()

if __name__ == '__main__':
    main()