from __future__ import print_function

import numpy as np
import re
from collections import OrderedDict

from examples.pybullet.utils.pybullet_tools.pr2_problems import create_pr2, create_table, Problem, get_pose
from examples.pybullet.utils.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, \
    arm_conf, REST_LEFT_ARM, close_arm, set_group_conf
from examples.pybullet.utils.pybullet_tools.utils import get_bodies, sample_placement, sample_placement_half, pairwise_collision, define_placement, \
    add_data_path, load_pybullet, set_point, Point, create_box, stable_z, joint_from_name, get_point, wait_for_user,\
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, YELLOW, CYAN, MAGENTA
from examples.pybullet.utils.pybullet_tools.utils import connect, get_pose, get_aabb, set_pose, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user
import os, json

AVAILABLE_COLORS = [
    RED, GREEN, BLUE, BLACK, WHITE,BROWN, TAN, GREY, YELLOW, CYAN, MAGENTA
]

COLOR_CONST = {
    'red':    RED,
    'green':  GREEN,
    'blue':   BLUE,
    'black':  BLACK,
    'white':  WHITE,
    'brown':  BROWN,
    'tan':    TAN,
    'grey':   GREY,
    'yellow': YELLOW,
    'cyan':   CYAN,
    'magenta': MAGENTA,
}


def parse_all_block_names(pddl_path):
    text = open(pddl_path).read()

    # a) :objects
    m = re.search(r'\(:objects\s*(.*?)\)', text, re.DOTALL|re.IGNORECASE)
    objs = set(re.findall(r'\b\w+\b', m.group(1))) if m else set()

    # b) :init 의 on/on-table
    init = text.split('(:init',1)[1].split('(:goal',1)[0]
    for a,b in re.findall(r'\(on\s+(\w+)\s+(\w+)\)', init):
        objs.add(a); objs.add(b)
    for a in re.findall(r'\(on-table\s+(\w+)\)', init):
        objs.add(a)

    # c) :goal 의 on/on-table
    goal = re.search(r'\(:goal\s*\(\s*and(.*?)\)\s*\)', text, re.DOTALL|re.IGNORECASE)
    if goal:
        gb = goal.group(1)
        for a,b in re.findall(r'\(on\s+(\w+)\s+(\w+)\)', gb):
            objs.add(a); objs.add(b)
        for a in re.findall(r'\(on-table\s+(\w+)\)', gb):
            objs.add(a)

    return sorted(objs)   # 순서를 보장하려면 정렬


# 2) 블록 생성 함수 수정
def create_blocks_from_pddl(pddl_path, block_width=0.08, block_height=0.08):
    names = parse_all_block_names(pddl_path)
    blocks_by_color = {}
    for name in names:
        lname = name.lower()
        if "black" in lname:
            color = COLOR_CONST["black"]
        else:
            color = COLOR_CONST.get(lname)
        if color is None:
            raise ValueError(
                f"Unknown color for object '{name}'. "
                f"Expected one of {list(COLOR_CONST.keys())} or a 'blackN' distractor."
            )
        blocks_by_color[name] = create_box(
            block_width, block_width, block_height, color=color
        )
    return blocks_by_color


def num_from_pddl(problem_pddl_path):
    # "problem" 바로 뒤에 오는 숫자(들)를 정규표현식으로 추출
    match = re.search(r'problem(\d+)', problem_pddl_path) or re.search(r'blocksworld_pr(\d+)', problem_pddl_path)
    if match:
        num = int(match.group(1))
        print(f"Extracted num = {num}")  # 여기서는 3이 출력됩니다.
    else:
        raise ValueError(f"Could not find a number after 'problem' in {problem_pddl_path}")
    return num


def parse_initial_surfaces(pddl_path, blocks_by_color, table_body):
    text = open(pddl_path).read()
    init = text.split('(:init',1)[1].split('(:goal',1)[0]

    # parent 맵 구성
    parent = {}
    for c,s in re.findall(r'\(on\s+(\w+)\s+(\w+)\)', init):
        parent[c] = s
    for c in re.findall(r'\(on-table\s+(\w+)\)', init):
        parent[c] = 'table'

    # 높이 계산
    heights = {}
    def get_height(name):
        if parent[name] == 'table':
            heights[name] = 0          # ← 여기 저장 추가
            return 0
        if name in heights:
            return heights[name]
        h = get_height(parent[name]) + 1
        heights[name] = h
        return h

    # 모든 블록에 대해 재귀 높이 채우기
    for name in parent:
        get_height(name)

    # 정렬 후 OrderedDict 생성
    sorted_names = sorted(parent.keys(), key=lambda n: heights[n])
    mapping = OrderedDict()
    for name in sorted_names:
        bid = blocks_by_color[name]
        surf = parent[name]
        sid = table_body if surf == 'table' else blocks_by_color[surf]
        mapping[bid] = sid
    return mapping


def sample_placements(body_surfaces, obstacles=None, min_distances={}):
    if obstacles is None:
        obstacles = [body for body in get_bodies() if body not in body_surfaces]
    obstacles = list(obstacles)
    # TODO: max attempts here
    for body, surface in body_surfaces.items():
        min_distance = min_distances.get(body, 0.01)
        while True:
            pose = sample_placement(body, surface)
            if pose is None:
                return False
            if not any(pairwise_collision(body, obst, max_distance=min_distance)
                       for obst in obstacles if obst not in [body, surface]):
                obstacles.append(body)
                break
    return True


def sample_placements_from_problem(body_surfaces, table, obstacles=None, max_attempts=100, min_distances={}, **define_kwargs):
    """
    body_surfaces: {body_id: surface_body_id, ...}
      - body_id: 위에 놓을 블록
      - surface_body_id: 바로 아래 블록(또는 테이블)

    obstacles:    충돌 검사 대상이 될 초기 장애물 목록. None 이면 body_surfaces에
                  포함된 body_id들 및 surface_id들만 빼고 전부 장애물로 간주.

    define_kwargs: define_placement에게 넘길 추가 인자(epsilon, bottom_link 등).
    """
    # 1) 초기 장애물 설정 (body_surfaces에 나오는 것들은 빼고)
    if obstacles is None:
        obstacles = [b for b in get_bodies()
                     if b not in body_surfaces and b not in body_surfaces.values()]
    obstacles = list(obstacles)

    # 2) body_surfaces 순서(아래→위)대로 배치
    for body, surface in body_surfaces.items():
        # 1) Table 위 블록 → sample_placement 사용
        if surface == table:
            pose = None
            min_distance = min_distances.get(body, 0.01)
            for _ in range(max_attempts):
                pose = sample_placement_half(body, surface, right_half=True)
                if pose is None:
                    continue
                # table 위 이미 놓인 블록들과 충돌 검사
                if not any(pairwise_collision(body, obst, max_distance=min_distance)
                           for obst in obstacles if obst != body):
                    break
            else:
                raise RuntimeError(f"테이블 위 블록 배치 실패: body={body}")
        # 2) 블록 위 블록 → 정확히 center에 올리기
        else:
            pose = define_placement(body, surface, **define_kwargs)
        # 3) '겹침'(penetration)만 검사: surface(=바로 아래 블록)와 body는 건너뛴다
        for obst in obstacles:
            if obst in (body, surface):
                continue
            # pairwise_collision은 기본 max_distance=0 이므로
            # 실제 겹침이 있을 때만 True 반환
            if pairwise_collision(body, obst):
                raise RuntimeError(
                    f"Collision detected between body={body} and obst={obst}"
                )

        # 4) 정상적으로 배치됐으면, 다음 블록들을 위한 장애물 목록에 추가
        obstacles.append(body)

    return True



def create_goal_on_from_pddl(pddl_path, blocks_by_color, table_body):
    text = open(pddl_path).read()
    # 1) greedy 매칭으로 전체 and-block을 잡아낸다
    m = re.search(r'\(:goal\s*\(\s*and([\s\S]*)\)\s*\)', text,
                  re.IGNORECASE)
    if not m:
        raise ValueError(f"No :goal section found in {pddl_path}")
    goal_block = m.group(1)

    # 2) on X Y, on-table X 모두 찾기
    on_pairs = re.findall(r'\(on\s+(\w+)\s+(\w+)\)', goal_block)
    on_table = re.findall(r'\(on-table\s+(\w+)\)',       goal_block)

    # 3) child→parent 매핑
    parent = {}
    for child, surf in on_pairs:
        parent[child] = surf
    for child in on_table:
        parent[child] = 'table'

    # 4) 높이 계산 (재귀, 바닥은 height=0 저장)
    heights = {}
    def get_height(name):
        if parent[name] == 'table':
            heights[name] = 0
            return 0
        if name in heights:
            return heights[name]
        h = get_height(parent[name]) + 1
        heights[name] = h
        return h

    for name in parent:
        get_height(name)

    # 5) 바닥부터 위로 정렬
    sorted_children = sorted(heights, key=lambda n: heights[n])

    # 6) body_id, surface_id 튜플로 반환
    goal_on = []
    for child in sorted_children:
        surf_name = parent[child]
        parent_id = table_body if surf_name == 'table' else blocks_by_color[surf_name]
        goal_on.append((blocks_by_color[child], parent_id))
    return goal_on


#######################################################

def packed(meta_json, prob_num, prob_idx, trial, arm='left', grasp_type='top', used_json_poses=True):
    problem_pddl_path = f"/home/minseo/robot_ws/src/tamp_llm/experiments/blocksworld_pr/problem/blocksworld_pr{prob_num}_{prob_idx}.pddl"
    num = num_from_pddl(problem_pddl_path)

    # TODO: packing problem where you have to place in one direction
    base_extent = 5.0

    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    block_width = 0.08
    block_height = 0.08
    #block_height = 2*block_width
    block_area = block_width*block_width

    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    add_data_path()
    floor = load_pybullet("plane.urdf")
    pr2 = create_pr2(fixed_base=True)
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    set_group_conf(pr2, 'base', [-1.0, 0, 0]) # Be careful to not set the pr2's pose

    table = create_table()
    # surfaces = [table]

    # create block objects from problem pddl
    blocks_by_color = create_blocks_from_pddl(problem_pddl_path, block_width, block_height)
    print("blocks_by_color: ", blocks_by_color)

    surfaces = [table] + list(blocks_by_color.values())

    initial_surfaces = parse_initial_surfaces(
        problem_pddl_path,
        blocks_by_color,
        table
    )
    print("initial_surfaces: ", initial_surfaces)
    blocks = list(blocks_by_color.values())

    min_distances = {block: 0.1 for block in blocks}

    # ---- JSON에서 pose 로드하여 직접 배치 ----
    def _find_entry(meta, prob_num, prob_idx, trial):
        for e in meta:
            if e.get("num") == prob_num and e.get("index") == prob_idx and e.get("trial") == trial:
                return e
        return None

    if used_json_poses:
        print("using poses from json")
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entry = _find_entry(meta, prob_num, prob_idx, trial)
        print("entry: ", entry)

        if entry and "objects" in entry:
            missing = []
            for name, bid in blocks_by_color.items():
                blk = entry["objects"].get(name)
                if not blk or "pose" not in blk:
                    missing.append(name)
                    continue
                pos = tuple(blk["pose"]["position"])
                quat = tuple(blk["pose"]["quaternion"])
                set_pose(bid, (pos, quat))
                # print(f"pose of {bid}:", get_pose(bid))
    else:
        sample_placements_from_problem(initial_surfaces, table, min_distances=min_distances)

    goal_on = create_goal_on_from_pddl(problem_pddl_path,
                                       blocks_by_color,
                                       table)
    print("goal_on: ", goal_on)

    return Problem(robot=pr2, movable=blocks, arms=[arm], grasp_types=[grasp_type], surfaces=surfaces,
                   #goal_holding=[(arm, block) for block in blocks])
                   goal_on=goal_on, base_limits=base_limits), initial_surfaces, blocks_by_color

# def packed(arm='left', grasp_type='top', problem_pddl_path=None):
#     num = num_from_pddl(problem_pddl_path)
#
#     # TODO: packing problem where you have to place in one direction
#     base_extent = 5.0
#
#     base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
#     block_width = 0.05
#     block_height = 0.05
#     #block_height = 2*block_width
#     block_area = block_width*block_width
#
#     other_arm = get_other_arm(arm)
#     initial_conf = get_carry_conf(arm, grasp_type)
#
#     add_data_path()
#     floor = load_pybullet("plane.urdf")
#     pr2 = create_pr2(fixed_base=True)
#     set_arm_conf(pr2, arm, initial_conf)
#     open_arm(pr2, arm)
#     set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
#     close_arm(pr2, other_arm)
#     set_group_conf(pr2, 'base', [-1.0, 0, 0]) # Be careful to not set the pr2's pose
#
#     table = create_table()
#     # surfaces = [table]
#
#     # create block objects from problem pddl
#     blocks_by_color = create_blocks_from_pddl(problem_pddl_path, block_width, block_height)
#     print("blocks_by_color: ", blocks_by_color)
#
#     surfaces = [table] + list(blocks_by_color.values())
#
#     # problem.pddl에서, 블럭들이 쌓인 순서를 읽어서 저장한다.
#     # 원래 코드: initial_surfaces = {block: table for block in blocks}
#     # 예를 들어, initial_surfaces = {blue: table, green: blue, red: green, grey: table, yellow: grey}
#     # 이 리스트는 블럭이 쌓인 순서대로 (bottom부터 top 까지 )problem pddl엥서 읽어서 저장해야 함.
#     initial_surfaces = parse_initial_surfaces(
#         problem_pddl_path,
#         blocks_by_color,
#         table
#     )
#     print("initial_surfaces: ", initial_surfaces)
#     blocks = list(blocks_by_color.values())
#
#     min_distances = {block: 0.1 for block in blocks}
#     sample_placements_from_problem(initial_surfaces, table, min_distances=min_distances)
#
#     goal_on = create_goal_on_from_pddl(problem_pddl_path,
#                                        blocks_by_color,
#                                        table)
#     print("goal_on: ", goal_on)
#
#     return Problem(robot=pr2, movable=blocks, arms=[arm], grasp_types=[grasp_type], surfaces=surfaces,
#                    #goal_holding=[(arm, block) for block in blocks])
#                    goal_on=goal_on, base_limits=base_limits), initial_surfaces, blocks_by_color

#######################################################

def blocked(arm='left', grasp_type='side', num=5):
    x_extent = 10.0

    base_limits = (-x_extent/2.*np.ones(2), x_extent/2.*np.ones(2))
    block_width = 0.07
    #block_height = 0.1
    block_height = 2*block_width
    #block_height = 0.2
    plate_height = 0.001
    table_x = (x_extent - 1) / 2.

    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    add_data_path()
    floor = load_pybullet("plane.urdf")
    pr2 = create_pr2()
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    set_group_conf(pr2, 'base', [x_extent/4, 0, 0]) # Be careful to not set the pr2's pose

    table1 = create_table()
    set_point(table1, Point(x=+table_x, y=0))
    table2 = create_table()
    set_point(table2, Point(x=-table_x, y=0))
    #table3 = create_table()
    #set_point(table3, Point(x=0, y=0))

    plate = create_box(0.6, 0.6, plate_height, color=GREEN)
    x, y, _ = get_point(table1)
    plate_z = stable_z(plate, table1)
    set_point(plate, Point(x=x, y=y-0.3, z=plate_z))
    #surfaces = [table1, table2, table3, plate]
    surfaces = [table1, table2, plate]

    green1 = create_box(block_width, block_width, block_height, color=BLUE)
    green1_z = stable_z(green1, table1)
    set_point(green1, Point(x=x, y=y+0.3, z=green1_z))
    # TODO: can consider a fixed wall here instead

    spacing = 0.15

    #red_directions = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
    red_directions = [(-1, 0)]
    #red_directions = []
    red_bodies = []
    for red_direction in red_directions:
        red = create_box(block_width, block_width, block_height, color=RED)
        red_bodies.append(red)
        x, y = get_point(green1)[:2] + spacing*np.array(red_direction)
        z = stable_z(red, table1)
        set_point(red, Point(x=x, y=y, z=z))

    wall1 = create_box(0.01, 2*spacing, block_height, color=GREY)
    wall2 = create_box(spacing, 0.01, block_height, color=GREY)
    wall3 = create_box(spacing, 0.01, block_height, color=GREY)
    z = stable_z(wall1, table1)
    x, y = get_point(green1)[:2]
    set_point(wall1, Point(x=x+spacing, y=y, z=z))
    set_point(wall2, Point(x=x+spacing/2, y=y+spacing, z=z))
    set_point(wall3, Point(x=x+spacing/2, y=y-spacing, z=z))

    green_bodies = [create_box(block_width, block_width, block_height, color=BLUE) for _ in range(num)]
    body_types = [(b, 'green') for b in [green1] + green_bodies] #  + [(table1, 'sink')]

    movable = [green1] + green_bodies + red_bodies
    initial_surfaces = {block: table2 for block in green_bodies}
    sample_placements(initial_surfaces)

    return Problem(robot=pr2, movable=movable, arms=[arm], grasp_types=[grasp_type], surfaces=surfaces,
                   #sinks=[table1],
                   #goal_holding=[(arm, '?green')],
                   #goal_cleaned=['?green'],
                   goal_on=[('?green', plate)],
                   body_types=body_types, base_limits=base_limits, costs=True)

#######################################################


PROBLEMS = [
    packed,
    blocked,
]


if __name__ == "__main__":
    import os
    import json

    # 결과가 저장될 JSON 경로
    out_json = "/home/minseo/robot_ws/src/tamp_llm/experiments/blocksworld_pr/problem/problems_meta_5_6.json"

    entries = []
    for num in [5]:
        for i in [6]:
            for j in [1, 2]:
                pddl_path = f"/home/minseo/robot_ws/src/tamp_llm/experiments/blocksworld_pr/problem/blocksworld_pr{num}_{i}.pddl"

                connect(use_gui=True)
                with HideOutput():
                    problem, init_surfaces, blocks_by_color = packed(meta_json = out_json, prob_num=num, prob_idx=i, trial=j, arm='left', grasp_type='top', used_json_poses=False)
                table_body = problem.surfaces[0]
                name_by_body = {bid: name for name, bid in blocks_by_color.items()}
                blocks = list(blocks_by_color.values())

                # 색(이름) -> {bid, pose} 메타데이터로 통합 저장
                blocks_meta = {}
                for name, bid in blocks_by_color.items():
                    pos, quat = get_pose(bid)  # ((x,y,z), (qx,qy,qz,qw))
                    lower, upper = get_aabb(bid)
                    size = [upper[2]-lower[2], upper[2]-lower[2], upper[2]-lower[2]]
                    blocks_meta[name] = {
                        "bid": int(bid),
                        "size": list(size),
                        "pose": {
                            "position": list(pos),
                            "quaternion": list(quat),
                        }
                    }

                # 초기 스택: 이름 기반 매핑
                init_surfaces_names = {}
                for child, parent in init_surfaces.items():
                    child_name = name_by_body.get(child, f"id{child}")
                    parent_name = "table" if parent == table_body else name_by_body.get(parent, f"id{parent}")
                    init_surfaces_names[child_name] = parent_name

                # 목표 스택: 이름 기반 매핑
                goal_on_names = {}
                for child, parent in problem.goal_on:
                    child_name = name_by_body.get(child, f"id{child}")
                    parent_name = "table" if parent == table_body else name_by_body.get(parent, f"id{parent}")
                    goal_on_names[child_name] = parent_name

                entry = {
                    "problem_name": f"blocksworld_pr{num}_{i}",
                    "pddl_path": pddl_path,
                    "num": num,
                    "index": i,
                    "trial": j,
                    "table_body": int(table_body),

                    # ✅ 통합 저장: color/name -> {bid, pose}
                    "objects": blocks_meta,

                    # 편의상 이름 기반으로만 저장(필요하면 id 버전도 추가 가능)
                    "init_surfaces": init_surfaces_names,
                    "goal_on": goal_on_names,
                }
                entries.append(entry)
                print(f"[ok] {entry['problem_name']}")

                # wait_for_user()
                disconnect()

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(entries)} entries to {out_json}")