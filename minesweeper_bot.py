"""
Minesweeper Bot — Windows 11 Microsoft Minesweeper 自动化
适用于高级模式 30x16, 99 颗雷
"""

import time
import sys
import itertools
from collections import defaultdict

import pyautogui
import numpy as np
from PIL import ImageGrab

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ROWS, COLS, MINES = 16, 30, 99

# 格子状态
UNKNOWN = -1
FLAGGED = -2
EMPTY = 0  # 已打开，周围无雷

# 颜色阈值 (RGB) — 针对 Microsoft Minesweeper 默认主题
# 未打开格子的两种绿色（深/浅交替棋盘格）
GREEN_DARK = np.array([74, 117, 44])
GREEN_LIGHT = np.array([132, 175, 91])
GREEN_TOLERANCE = 35

# 已打开空白格的棕色
BROWN_COLORS = [
    np.array([215, 184, 153]),
    np.array([229, 194, 159]),
    np.array([187, 160, 125]),
    np.array([200, 175, 145]),
]
BROWN_TOLERANCE = 40

# 数字颜色 (中心像素主色调)
NUMBER_COLORS = {
    1: (25, 118, 210),    # 蓝色
    2: (56, 142, 60),     # 绿色
    3: (211, 47, 47),     # 红色
    4: (123, 31, 162),    # 紫色
    5: (255, 143, 0),     # 橙色
    6: (0, 151, 167),     # 青色
    7: (66, 66, 66),      # 深灰
    8: (158, 158, 158),   # 浅灰
}
NUMBER_TOLERANCE = 55

# 旗帜检测：红色像素占比阈值
FLAG_RED_RATIO = 0.15

# pyautogui 设置
pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True  # 鼠标移到左上角可中止


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def color_distance(c1, c2):
    """欧氏距离"""
    return np.sqrt(np.sum((np.array(c1, dtype=float) - np.array(c2, dtype=float)) ** 2))


def is_green(pixel):
    """判断像素是否为未打开格子的绿色"""
    return (color_distance(pixel, GREEN_DARK) < GREEN_TOLERANCE or
            color_distance(pixel, GREEN_LIGHT) < GREEN_TOLERANCE)


def is_brown(pixel):
    """判断像素是否为已打开空白格的棕色"""
    return any(color_distance(pixel, b) < BROWN_TOLERANCE for b in BROWN_COLORS)


def is_red(pixel):
    """判断像素是否为红色（旗帜）"""
    r, g, b = pixel[0], pixel[1], pixel[2]
    return r > 180 and g < 100 and b < 100


# ---------------------------------------------------------------------------
# 棋盘定位
# ---------------------------------------------------------------------------
def locate_board(screenshot):
    """
    在截图中定位扫雷棋盘。
    通过扫描绿色像素的连续区域来找到棋盘边界，
    然后推算格子大小和左上角坐标。
    返回 (board_x, board_y, cell_size) 或 None。
    """
    img = np.array(screenshot)
    h, w = img.shape[:2]

    # 创建绿色掩码
    green_mask = np.zeros((h, w), dtype=bool)
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            pixel = img[y, x]
            if is_green(pixel):
                green_mask[y, x] = True
                # 填充邻近像素
                if y + 1 < h:
                    green_mask[y + 1, x] = True
                if x + 1 < w:
                    green_mask[y, x + 1] = True
                    if y + 1 < h:
                        green_mask[y + 1, x + 1] = True

    # 找绿色区域的边界
    green_rows = np.any(green_mask, axis=1)
    green_cols = np.any(green_mask, axis=0)

    if not np.any(green_rows) or not np.any(green_cols):
        return None

    top = np.argmax(green_rows)
    bottom = h - 1 - np.argmax(green_rows[::-1])
    left = np.argmax(green_cols)
    right = w - 1 - np.argmax(green_cols[::-1])

    board_width = right - left + 1
    board_height = bottom - top + 1

    # 推算格子大小
    cell_w = board_width / COLS
    cell_h = board_height / ROWS

    # 格子应该是正方形或接近正方形
    cell_size = round((cell_w + cell_h) / 2)

    if cell_size < 10 or cell_size > 100:
        return None

    # 微调左上角：确保对齐到第一个格子中心是绿色
    board_x = left
    board_y = top

    return board_x, board_y, cell_size


def fast_locate_board(screenshot):
    """
    快速定位：先缩小图片扫描，再精确定位。
    """
    img = np.array(screenshot)
    h, w = img.shape[:2]

    # 缩小 4 倍扫描
    scale = 4
    small_h, small_w = h // scale, w // scale

    green_xs = []
    green_ys = []

    for sy in range(small_h):
        for sx in range(small_w):
            y, x = sy * scale, sx * scale
            pixel = img[y, x]
            if is_green(pixel):
                green_xs.append(x)
                green_ys.append(y)

    if len(green_xs) < 50:
        return None

    # 粗略边界
    rough_left = min(green_xs)
    rough_right = max(green_xs)
    rough_top = min(green_ys)
    rough_bottom = max(green_ys)

    # 在粗略边界附近精确扫描
    margin = scale * 2
    fine_left = max(0, rough_left - margin)
    fine_right = min(w - 1, rough_right + margin)
    fine_top = max(0, rough_top - margin)
    fine_bottom = min(h - 1, rough_bottom + margin)

    exact_left, exact_right = fine_right, fine_left
    exact_top, exact_bottom = fine_bottom, fine_top

    for y in range(fine_top, fine_bottom + 1):
        for x in range(fine_left, fine_right + 1):
            if is_green(img[y, x]):
                exact_left = min(exact_left, x)
                exact_right = max(exact_right, x)
                exact_top = min(exact_top, y)
                exact_bottom = max(exact_bottom, y)

    board_width = exact_right - exact_left + 1
    board_height = exact_bottom - exact_top + 1

    cell_w = board_width / COLS
    cell_h = board_height / ROWS
    cell_size = round((cell_w + cell_h) / 2)

    if cell_size < 10 or cell_size > 100:
        return None

    return exact_left, exact_top, cell_size


# ---------------------------------------------------------------------------
# 格子识别
# ---------------------------------------------------------------------------
def get_cell_center(board_x, board_y, cell_size, row, col):
    """获取格子 (row, col) 的屏幕中心坐标"""
    cx = board_x + col * cell_size + cell_size // 2
    cy = board_y + row * cell_size + cell_size // 2
    return cx, cy


def identify_cell(img_array, board_x, board_y, cell_size, row, col):
    """
    识别单个格子的状态。
    返回: UNKNOWN, FLAGGED, EMPTY, 或 1-8
    """
    cx, cy = get_cell_center(board_x, board_y, cell_size, row, col)

    # 采样区域：格子中心附近的像素
    sample_radius = max(2, cell_size // 6)
    x1 = max(0, cx - sample_radius)
    y1 = max(0, cy - sample_radius)
    x2 = min(img_array.shape[1] - 1, cx + sample_radius)
    y2 = min(img_array.shape[0] - 1, cy + sample_radius)

    region = img_array[y1:y2 + 1, x1:x2 + 1]
    if region.size == 0:
        return UNKNOWN

    avg_color = region.mean(axis=(0, 1))

    # 1. 检查是否为未打开（绿色）
    if is_green(avg_color):
        # 进一步检查是否有旗帜（红色像素占比）
        red_count = 0
        total = region.shape[0] * region.shape[1]
        for py in range(region.shape[0]):
            for px in range(region.shape[1]):
                if is_red(region[py, px]):
                    red_count += 1
        if total > 0 and red_count / total > FLAG_RED_RATIO:
            return FLAGGED
        return UNKNOWN

    # 2. 检查是否为已打开空白
    if is_brown(avg_color):
        # 可能是空白，也可能有数字 — 检查中心像素颜色
        center_pixel = img_array[cy, cx]

        # 如果中心像素也是棕色，则为空白
        if is_brown(center_pixel):
            return EMPTY

        # 否则尝试匹配数字颜色
        best_num = None
        best_dist = NUMBER_TOLERANCE
        for num, nc in NUMBER_COLORS.items():
            d = color_distance(center_pixel, nc)
            if d < best_dist:
                best_dist = d
                best_num = num

        if best_num is not None:
            return best_num

        # 扩大采样范围再试一次
        sample_r2 = max(3, cell_size // 4)
        sx1 = max(0, cx - sample_r2)
        sy1 = max(0, cy - sample_r2)
        sx2 = min(img_array.shape[1] - 1, cx + sample_r2)
        sy2 = min(img_array.shape[0] - 1, cy + sample_r2)
        larger_region = img_array[sy1:sy2 + 1, sx1:sx2 + 1]

        # 找非棕色像素的主色调
        non_brown_pixels = []
        for py in range(larger_region.shape[0]):
            for px in range(larger_region.shape[1]):
                p = larger_region[py, px]
                if not is_brown(p):
                    non_brown_pixels.append(p)

        if non_brown_pixels:
            avg_non_brown = np.mean(non_brown_pixels, axis=0)
            best_num = None
            best_dist = NUMBER_TOLERANCE + 15
            for num, nc in NUMBER_COLORS.items():
                d = color_distance(avg_non_brown, nc)
                if d < best_dist:
                    best_dist = d
                    best_num = num
            if best_num is not None:
                return best_num

        return EMPTY

    # 3. 直接尝试匹配数字颜色（某些主题下背景不同）
    center_pixel = img_array[cy, cx]
    best_num = None
    best_dist = NUMBER_TOLERANCE
    for num, nc in NUMBER_COLORS.items():
        d = color_distance(center_pixel, nc)
        if d < best_dist:
            best_dist = d
            best_num = num
    if best_num is not None:
        return best_num

    # 默认当作已打开空白
    return EMPTY


def read_board(screenshot, board_x, board_y, cell_size):
    """读取整个棋盘状态，返回 ROWS x COLS 的二维列表"""
    img = np.array(screenshot)
    board = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            state = identify_cell(img, board_x, board_y, cell_size, r, c)
            row.append(state)
        board.append(row)
    return board


# ---------------------------------------------------------------------------
# 求解引擎
# ---------------------------------------------------------------------------
def get_neighbors(r, c):
    """获取 (r, c) 的所有有效邻居坐标"""
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                neighbors.append((nr, nc))
    return neighbors


def basic_solve(board):
    """
    基础规则求解。
    返回 (safe_cells, mine_cells):
      safe_cells: 确定安全可以点开的格子集合
      mine_cells: 确定是雷需要标旗的格子集合
    """
    safe = set()
    mines = set()

    for r in range(ROWS):
        for c in range(COLS):
            num = board[r][c]
            if num < 1 or num > 8:
                continue

            neighbors = get_neighbors(r, c)
            unknown = [(nr, nc) for nr, nc in neighbors if board[nr][nc] == UNKNOWN]
            flagged = [(nr, nc) for nr, nc in neighbors if board[nr][nc] == FLAGGED]

            remaining_mines = num - len(flagged)

            if remaining_mines == 0 and unknown:
                # 所有未知格都安全
                safe.update(unknown)
            elif remaining_mines == len(unknown) and unknown:
                # 所有未知格都是雷
                mines.update(unknown)

    return safe, mines


def constraint_solve(board):
    """
    约束推理求解：联立相邻数字格的约束关系。
    返回 (safe_cells, mine_cells)
    """
    safe = set()
    mines = set()

    # 收集所有约束: (未知格集合, 雷数)
    constraints = []
    for r in range(ROWS):
        for c in range(COLS):
            num = board[r][c]
            if num < 1 or num > 8:
                continue
            neighbors = get_neighbors(r, c)
            unknown = frozenset((nr, nc) for nr, nc in neighbors if board[nr][nc] == UNKNOWN)
            flagged_count = sum(1 for nr, nc in neighbors if board[nr][nc] == FLAGGED)
            remaining = num - flagged_count
            if unknown and 0 <= remaining <= len(unknown):
                constraints.append((unknown, remaining))

    # 两两比较约束，寻找子集关系
    changed = True
    iterations = 0
    while changed and iterations < 5:
        changed = False
        iterations += 1
        new_constraints = []

        for i, (s1, n1) in enumerate(constraints):
            for j, (s2, n2) in enumerate(constraints):
                if i >= j:
                    continue
                # 如果 s1 是 s2 的子集
                if s1 < s2:
                    diff = s2 - s1
                    diff_mines = n2 - n1
                    if diff_mines == 0:
                        safe.update(diff)
                        changed = True
                    elif diff_mines == len(diff):
                        mines.update(diff)
                        changed = True
                    elif 0 < diff_mines < len(diff):
                        new_constraints.append((frozenset(diff), diff_mines))
                # 如果 s2 是 s1 的子集
                elif s2 < s1:
                    diff = s1 - s2
                    diff_mines = n1 - n2
                    if diff_mines == 0:
                        safe.update(diff)
                        changed = True
                    elif diff_mines == len(diff):
                        mines.update(diff)
                        changed = True
                    elif 0 < diff_mines < len(diff):
                        new_constraints.append((frozenset(diff), diff_mines))

        # 合并新约束
        existing = set((s, n) for s, n in constraints)
        for nc in new_constraints:
            if nc not in existing:
                constraints.append(nc)
                existing.add(nc)

    return safe, mines


def probability_guess(board):
    """
    概率猜测：当没有确定解时，选择雷概率最低的格子。
    优先选择角落和边缘的格子。
    返回最佳猜测的 (row, col) 或 None。
    """
    # 收集所有未知格
    unknown_cells = []
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == UNKNOWN:
                unknown_cells.append((r, c))

    if not unknown_cells:
        return None

    # 计算每个未知格的"危险分数"
    # 分数越高越危险
    danger = defaultdict(float)
    constraint_count = defaultdict(int)

    for r in range(ROWS):
        for c in range(COLS):
            num = board[r][c]
            if num < 1 or num > 8:
                continue
            neighbors = get_neighbors(r, c)
            unknown = [(nr, nc) for nr, nc in neighbors if board[nr][nc] == UNKNOWN]
            flagged_count = sum(1 for nr, nc in neighbors if board[nr][nc] == FLAGGED)
            remaining = num - flagged_count

            if unknown and remaining > 0:
                prob = remaining / len(unknown)
                for cell in unknown:
                    danger[cell] += prob
                    constraint_count[cell] += 1

    # 对有约束信息的格子取平均概率
    avg_danger = {}
    for cell in unknown_cells:
        if cell in constraint_count and constraint_count[cell] > 0:
            avg_danger[cell] = danger[cell] / constraint_count[cell]
        else:
            # 没有约束信息的格子，用全局概率估算
            total_flags = sum(1 for r in range(ROWS) for c in range(COLS) if board[r][c] == FLAGGED)
            remaining_mines = MINES - total_flags
            remaining_unknown = len(unknown_cells)
            if remaining_unknown > 0:
                avg_danger[cell] = remaining_mines / remaining_unknown
            else:
                avg_danger[cell] = 0.5

    # 选择危险分数最低的格子
    # 同分时优先选角落 > 边缘 > 中间
    def priority(cell):
        r, c = cell
        d = avg_danger.get(cell, 0.5)
        # 角落优先级最高（值最小）
        is_corner = (r in (0, ROWS - 1)) and (c in (0, COLS - 1))
        is_edge = r in (0, ROWS - 1) or c in (0, COLS - 1)
        pos_bonus = 0
        if is_corner:
            pos_bonus = -0.05
        elif is_edge:
            pos_bonus = -0.02
        return d + pos_bonus

    best = min(unknown_cells, key=priority)
    return best


# ---------------------------------------------------------------------------
# 鼠标操作
# ---------------------------------------------------------------------------
def click_cell(board_x, board_y, cell_size, row, col, button='left'):
    """点击指定格子"""
    cx, cy = get_cell_center(board_x, board_y, cell_size, row, col)
    pyautogui.click(cx, cy, button=button)


def flag_cell(board_x, board_y, cell_size, row, col):
    """右键标旗"""
    click_cell(board_x, board_y, cell_size, row, col, button='right')


# ---------------------------------------------------------------------------
# 游戏状态检测
# ---------------------------------------------------------------------------
def count_states(board):
    """统计棋盘各状态数量"""
    unknown = 0
    flagged = 0
    opened = 0
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r][c]
            if v == UNKNOWN:
                unknown += 1
            elif v == FLAGGED:
                flagged += 1
            else:
                opened += 1
    return unknown, flagged, opened


def is_game_over(board, prev_board):
    """
    检测游戏是否结束：
    - 所有非雷格子都已打开 → 胜利
    - 棋盘状态突然大量变化（踩雷后所有雷显示）→ 失败
    """
    unknown, flagged, opened = count_states(board)

    # 胜利条件：未知格 + 旗帜 == 雷数
    if unknown + flagged == MINES:
        return True, "WIN"

    # 如果没有未知格了
    if unknown == 0:
        return True, "WIN"

    # 检测踩雷：如果上一轮和这一轮之间大量格子同时变化
    if prev_board is not None:
        changes = 0
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != prev_board[r][c]:
                    changes += 1
        # 踩雷后通常会有大量格子同时翻开（显示所有雷）
        if changes > ROWS * COLS * 0.3:
            return True, "LOSE"

    return False, None


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def print_board(board):
    """打印棋盘到控制台（调试用）"""
    symbols = {
        UNKNOWN: '.',
        FLAGGED: 'F',
        EMPTY: ' ',
    }
    print("   " + "".join(f"{c:2d}" for c in range(COLS)))
    print("   " + "--" * COLS)
    for r in range(ROWS):
        row_str = f"{r:2d}|"
        for c in range(COLS):
            v = board[r][c]
            if v in symbols:
                row_str += f" {symbols[v]}"
            else:
                row_str += f" {v}"
        print(row_str)


def main():
    print("=" * 50)
    print("  扫雷自动化机器人 (Minesweeper Bot)")
    print("  适用于 Windows 11 Microsoft Minesweeper")
    print("  高级模式 30x16, 99 颗雷")
    print("=" * 50)
    print()
    print("请先打开 Microsoft Minesweeper 并开始一局高级模式游戏。")
    print("程序将在 3 秒后开始，请切换到扫雷窗口...")
    print()

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  开始!")
    print()

    # 截图并定位棋盘
    print("[1] 截图定位棋盘...")
    screenshot = ImageGrab.grab()
    board_info = fast_locate_board(screenshot)

    if board_info is None:
        print("错误：无法定位扫雷棋盘！")
        print("请确保：")
        print("  1. Microsoft Minesweeper 已打开且可见")
        print("  2. 正在进行高级模式 (30x16) 游戏")
        print("  3. 棋盘没有被其他窗口遮挡")
        input("按回车键退出...")
        sys.exit(1)

    board_x, board_y, cell_size = board_info
    print(f"  棋盘位置: ({board_x}, {board_y}), 格子大小: {cell_size}px")

    # 第一次点击：棋盘中心
    print("[2] 第一次点击（棋盘中心）...")
    center_r, center_c = ROWS // 2, COLS // 2
    click_cell(board_x, board_y, cell_size, center_r, center_c)
    time.sleep(0.5)

    prev_board = None
    turn = 0
    stale_count = 0
    max_stale = 3

    print("[3] 开始自动求解循环...")
    print()

    while True:
        turn += 1
        print(f"--- 第 {turn} 轮 ---")

        # 截图识别
        time.sleep(0.15)
        screenshot = ImageGrab.grab()
        board = read_board(screenshot, board_x, board_y, cell_size)

        unknown, flagged, opened = count_states(board)
        print(f"  未知: {unknown}, 旗帜: {flagged}, 已开: {opened}")

        # 检测游戏结束
        game_over, result = is_game_over(board, prev_board)
        if game_over:
            print()
            if result == "WIN":
                print("*** 游戏胜利! ***")
            else:
                print("*** 踩雷了，游戏结束 ***")
            break

        # 求解
        safe, mines_found = basic_solve(board)

        if not safe and not mines_found:
            # 基础规则无解，尝试约束推理
            safe, mines_found = constraint_solve(board)

        actions = 0

        # 标旗
        for r, c in mines_found:
            if board[r][c] == UNKNOWN:
                flag_cell(board_x, board_y, cell_size, r, c)
                board[r][c] = FLAGGED
                actions += 1

        # 点开安全格
        for r, c in safe:
            if board[r][c] == UNKNOWN:
                click_cell(board_x, board_y, cell_size, r, c)
                actions += 1

        if actions == 0:
            # 没有确定解，概率猜测
            guess = probability_guess(board)
            if guess is None:
                print("  没有可操作的格子，结束。")
                break
            r, c = guess
            print(f"  概率猜测: ({r}, {c})")
            click_cell(board_x, board_y, cell_size, r, c)
            actions = 1
            stale_count = 0
        else:
            print(f"  执行了 {actions} 个操作 (安全: {len(safe)}, 标雷: {len(mines_found)})")

        # 检测是否卡住（连续多轮无变化）
        if prev_board is not None and board == prev_board:
            stale_count += 1
            if stale_count >= max_stale:
                print("  棋盘状态连续无变化，尝试重新截图...")
                time.sleep(0.5)
                stale_count = 0
        else:
            stale_count = 0

        prev_board = [row[:] for row in board]

        # 短暂等待让游戏动画完成
        time.sleep(0.1)

    print()
    print("程序结束。")
    if sys.platform == 'win32':
        input("按回车键退出...")


if __name__ == "__main__":
    main()
