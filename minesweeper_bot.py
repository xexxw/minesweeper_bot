"""
Minesweeper Bot — Windows 11 Microsoft Minesweeper 自动化
适用于高级模式 30x16, 99 颗雷

快捷键:
  F9  — 暂停 / 继续
  F10 — 停止退出
"""

import time
import sys
import threading
import itertools
from collections import defaultdict

import pyautogui
import numpy as np
from PIL import ImageGrab

try:
    import keyboard  # 用于全局快捷键
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ROWS, COLS, MINES = 16, 30, 99

# 格子状态
UNKNOWN = -1
FLAGGED = -2
EMPTY = 0  # 已打开，周围无雷

# 颜色阈值 (RGB) — 针对 Microsoft Minesweeper 默认主题
GREEN_DARK = np.array([74, 117, 44])
GREEN_LIGHT = np.array([132, 175, 91])
GREEN_TOLERANCE = 35

BROWN_COLORS = [
    np.array([215, 184, 153]),
    np.array([229, 194, 159]),
    np.array([187, 160, 125]),
    np.array([200, 175, 145]),
]
BROWN_TOLERANCE = 40

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

FLAG_RED_RATIO = 0.15

# pyautogui 设置
pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

# ---------------------------------------------------------------------------
# 全局控制状态
# ---------------------------------------------------------------------------
paused = False
stopped = False


def on_pause():
    global paused
    paused = not paused
    if paused:
        print("\n*** 已暂停 (按 F9 继续) ***")
    else:
        print("*** 继续运行 ***")


def on_stop():
    global stopped
    stopped = True
    print("\n*** 正在停止... ***")


def setup_hotkeys():
    if HAS_KEYBOARD:
        keyboard.add_hotkey('F9', on_pause)
        keyboard.add_hotkey('F10', on_stop)
        print("快捷键已注册: F9=暂停/继续, F10=停止")
    else:
        print("提示: 安装 keyboard 库可启用快捷键 (pip install keyboard)")
        print("当前可通过将鼠标移到屏幕左上角来紧急停止")


def wait_if_paused():
    """如果暂停了就等待"""
    while paused and not stopped:
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def color_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1, dtype=float) - np.array(c2, dtype=float)) ** 2))


def is_green(pixel):
    return (color_distance(pixel, GREEN_DARK) < GREEN_TOLERANCE or
            color_distance(pixel, GREEN_LIGHT) < GREEN_TOLERANCE)


def is_brown(pixel):
    return any(color_distance(pixel, b) < BROWN_TOLERANCE for b in BROWN_COLORS)


def is_red(pixel):
    r, g, b = pixel[0], pixel[1], pixel[2]
    return r > 180 and g < 100 and b < 100


# ---------------------------------------------------------------------------
# 棋盘定位
# ---------------------------------------------------------------------------
def fast_locate_board(screenshot):
    """
    通过网格密度分析定位棋盘。
    扫雷棋盘的特征：绿色像素呈规则网格排列，密度高且均匀。
    背景中的绿色（树木、壁纸等）密度不均匀。
    """
    img = np.array(screenshot)
    h, w = img.shape[:2]

    # 第一步：缩小扫描，建立绿色像素网格
    scale = 4
    small_h, small_w = h // scale, w // scale

    # 用二维数组记录每个缩小格子是否为绿色
    green_grid = np.zeros((small_h, small_w), dtype=bool)

    for sy in range(small_h):
        for sx in range(small_w):
            y, x = sy * scale, sx * scale
            if is_green(img[y, x]):
                green_grid[sy, sx] = True

    if np.sum(green_grid) < 30:
        return None

    # 第二步：用滑动窗口找绿色密度最高的矩形区域
    # 扫雷棋盘是 30:16 的宽高比 ≈ 1.875
    # 在缩小后的图上，用行列密度分析找棋盘
    # 对每一行统计绿色像素数
    row_density = np.sum(green_grid, axis=1)
    col_density = np.sum(green_grid, axis=0)

    # 找行密度高的连续区域（棋盘所在的行）
    row_threshold = small_w * 0.05  # 一行至少 5% 是绿色
    col_threshold = small_h * 0.05

    dense_rows = row_density > row_threshold
    dense_cols = col_density > col_threshold

    if not np.any(dense_rows) or not np.any(dense_cols):
        return None

    # 找最长的连续密集行段
    def find_longest_run(arr):
        best_start, best_len = 0, 0
        start, length = 0, 0
        for i, v in enumerate(arr):
            if v:
                if length == 0:
                    start = i
                length += 1
                if length > best_len:
                    best_start, best_len = start, length
            else:
                length = 0
        return best_start, best_len

    row_start, row_len = find_longest_run(dense_rows)
    col_start, col_len = find_longest_run(dense_cols)

    if row_len < 5 or col_len < 5:
        return None

    # 第三步：在候选区域内精确定位
    # 只看密集区域内的绿色像素
    candidate_top = row_start * scale
    candidate_bottom = min(h - 1, (row_start + row_len) * scale)
    candidate_left = col_start * scale
    candidate_right = min(w - 1, (col_start + col_len) * scale)

    # 在候选区域内进一步验证：检查绿色像素是否呈网格状
    # 棋盘的绿色格子是交替的棋盘格，所以绿色像素应该有规律的间隔
    exact_left, exact_right = candidate_right, candidate_left
    exact_top, exact_bottom = candidate_bottom, candidate_top

    for y in range(candidate_top, candidate_bottom + 1, 2):
        for x in range(candidate_left, candidate_right + 1, 2):
            if is_green(img[y, x]):
                exact_left = min(exact_left, x)
                exact_right = max(exact_right, x)
                exact_top = min(exact_top, y)
                exact_bottom = max(exact_bottom, y)

    board_width = exact_right - exact_left + 1
    board_height = exact_bottom - exact_top + 1

    # 验证宽高比：30:16 ≈ 1.875，允许一定误差
    ratio = board_width / max(board_height, 1)
    expected_ratio = COLS / ROWS  # 1.875
    if abs(ratio - expected_ratio) > 0.4:
        # 宽高比不对，可能包含了背景绿色
        # 尝试用宽高比修正：假设宽度正确，修正高度（或反之）
        if ratio > expected_ratio:
            # 太宽了，可能左右包含了背景
            # 用高度推算正确宽度
            corrected_width = int(board_height * expected_ratio)
            center_x = (exact_left + exact_right) // 2
            exact_left = center_x - corrected_width // 2
            exact_right = exact_left + corrected_width
        else:
            # 太高了
            corrected_height = int(board_width / expected_ratio)
            center_y = (exact_top + exact_bottom) // 2
            exact_top = center_y - corrected_height // 2
            exact_bottom = exact_top + corrected_height

        board_width = exact_right - exact_left + 1
        board_height = exact_bottom - exact_top + 1

    cell_w = board_width / COLS
    cell_h = board_height / ROWS
    cell_size = round((cell_w + cell_h) / 2)

    if cell_size < 10 or cell_size > 100:
        return None

    return exact_left, exact_top, cell_size


def save_debug_image(screenshot, board_x, board_y, cell_size):
    """保存调试图片，标注识别到的棋盘网格"""
    from PIL import ImageDraw
    debug_img = screenshot.copy()
    draw = ImageDraw.Draw(debug_img)

    # 画棋盘边框（红色）
    bw = cell_size * COLS
    bh = cell_size * ROWS
    draw.rectangle([board_x, board_y, board_x + bw, board_y + bh], outline='red', width=3)

    # 画网格线（黄色）
    for r in range(ROWS + 1):
        y = board_y + r * cell_size
        draw.line([(board_x, y), (board_x + bw, y)], fill='yellow', width=1)
    for c in range(COLS + 1):
        x = board_x + c * cell_size
        draw.line([(x, board_y), (x, board_y + bh)], fill='yellow', width=1)

    import os
    debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_board.png")
    debug_img.save(debug_path)
    print(f"  调试图片已保存: {debug_path}")
    return debug_path


# ---------------------------------------------------------------------------
# 格子识别
# ---------------------------------------------------------------------------
def get_cell_center(board_x, board_y, cell_size, row, col):
    cx = board_x + col * cell_size + cell_size // 2
    cy = board_y + row * cell_size + cell_size // 2
    return cx, cy


def identify_cell(img_array, board_x, board_y, cell_size, row, col):
    cx, cy = get_cell_center(board_x, board_y, cell_size, row, col)

    sample_radius = max(2, cell_size // 6)
    x1 = max(0, cx - sample_radius)
    y1 = max(0, cy - sample_radius)
    x2 = min(img_array.shape[1] - 1, cx + sample_radius)
    y2 = min(img_array.shape[0] - 1, cy + sample_radius)

    region = img_array[y1:y2 + 1, x1:x2 + 1]
    if region.size == 0:
        return UNKNOWN

    avg_color = region.mean(axis=(0, 1))

    # 1. 未打开（绿色）
    if is_green(avg_color):
        # 检查旗帜
        red_count = 0
        total = region.shape[0] * region.shape[1]
        for py in range(region.shape[0]):
            for px in range(region.shape[1]):
                if is_red(region[py, px]):
                    red_count += 1
        if total > 0 and red_count / total > FLAG_RED_RATIO:
            return FLAGGED
        return UNKNOWN

    # 2. 已打开
    if is_brown(avg_color):
        center_pixel = img_array[cy, cx]

        if is_brown(center_pixel):
            return EMPTY

        best_num = None
        best_dist = NUMBER_TOLERANCE
        for num, nc in NUMBER_COLORS.items():
            d = color_distance(center_pixel, nc)
            if d < best_dist:
                best_dist = d
                best_num = num

        if best_num is not None:
            return best_num

        # 扩大采样
        sample_r2 = max(3, cell_size // 4)
        sx1 = max(0, cx - sample_r2)
        sy1 = max(0, cy - sample_r2)
        sx2 = min(img_array.shape[1] - 1, cx + sample_r2)
        sy2 = min(img_array.shape[0] - 1, cy + sample_r2)
        larger_region = img_array[sy1:sy2 + 1, sx1:sx2 + 1]

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

    # 3. 其他颜色 — 尝试匹配数字
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

    return UNKNOWN


def read_board(screenshot, board_x, board_y, cell_size, known_flags=None):
    """
    读取整个棋盘状态。
    known_flags: 已知的旗帜位置集合，用于补充截图识别不到的旗帜。
    """
    img = np.array(screenshot)
    board = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            val = identify_cell(img, board_x, board_y, cell_size, r, c)
            # 如果内存中记录了这个格子是旗帜，但截图没识别出来，保留旗帜状态
            if known_flags and (r, c) in known_flags and val == UNKNOWN:
                val = FLAGGED
            row.append(val)
        board.append(row)
    return board


# ---------------------------------------------------------------------------
# 求解引擎
# ---------------------------------------------------------------------------
def get_neighbors(r, c):
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
                safe.update(unknown)
            elif remaining_mines == len(unknown) and unknown:
                mines.update(unknown)

    return safe, mines


def constraint_solve(board):
    safe = set()
    mines = set()

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

        existing = set((s, n) for s, n in constraints)
        for nc in new_constraints:
            if nc not in existing:
                constraints.append(nc)
                existing.add(nc)

    return safe, mines


def probability_guess(board):
    unknown_cells = []
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == UNKNOWN:
                unknown_cells.append((r, c))

    if not unknown_cells:
        return None

    # 优先选靠近已开区域边缘的格子（有数字邻居的）
    frontier = []
    interior = []
    for (r, c) in unknown_cells:
        has_opened_neighbor = False
        for nr, nc in get_neighbors(r, c):
            if board[nr][nc] >= 0:
                has_opened_neighbor = True
                break
        if has_opened_neighbor:
            frontier.append((r, c))
        else:
            interior.append((r, c))

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

    avg_danger = {}
    total_flags = sum(1 for r in range(ROWS) for c in range(COLS) if board[r][c] == FLAGGED)
    remaining_mines = MINES - total_flags
    remaining_unknown = len(unknown_cells)
    global_prob = remaining_mines / remaining_unknown if remaining_unknown > 0 else 0.5

    for cell in unknown_cells:
        if cell in constraint_count and constraint_count[cell] > 0:
            avg_danger[cell] = danger[cell] / constraint_count[cell]
        else:
            avg_danger[cell] = global_prob

    def priority(cell):
        r, c = cell
        d = avg_danger.get(cell, 0.5)
        is_corner = (r in (0, ROWS - 1)) and (c in (0, COLS - 1))
        is_edge = r in (0, ROWS - 1) or c in (0, COLS - 1)
        pos_bonus = 0
        if is_corner:
            pos_bonus = -0.05
        elif is_edge:
            pos_bonus = -0.02
        return d + pos_bonus

    # 优先从 frontier 中选（信息更多），如果 frontier 都很危险就选 interior
    candidates = frontier if frontier else interior
    if not candidates:
        candidates = unknown_cells

    best = min(candidates, key=priority)
    return best


# ---------------------------------------------------------------------------
# 鼠标操作
# ---------------------------------------------------------------------------
def click_cell(board_x, board_y, cell_size, row, col, button='left'):
    cx, cy = get_cell_center(board_x, board_y, cell_size, row, col)
    pyautogui.click(cx, cy, button=button)


def flag_cell(board_x, board_y, cell_size, row, col):
    click_cell(board_x, board_y, cell_size, row, col, button='right')


# ---------------------------------------------------------------------------
# 游戏状态检测
# ---------------------------------------------------------------------------
def count_states(board):
    unknown = flagged = opened = 0
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
    unknown, flagged, opened = count_states(board)

    if unknown + flagged == MINES:
        return True, "WIN"
    if unknown == 0:
        return True, "WIN"

    if prev_board is not None:
        changes = 0
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != prev_board[r][c]:
                    changes += 1
        if changes > ROWS * COLS * 0.3:
            return True, "LOSE"

    return False, None


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def print_board(board):
    symbols = {UNKNOWN: '.', FLAGGED: 'F', EMPTY: ' '}
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
    global stopped

    print("=" * 50)
    print("  扫雷自动化机器人 Minesweeper Bot")
    print("  高级模式 30x16, 99 颗雷")
    print("=" * 50)
    print()

    setup_hotkeys()
    print()
    print("请先打开 Microsoft Minesweeper 并开始一局高级模式游戏。")
    print("确保扫雷窗口完整可见，不要被遮挡。")
    print()
    input("准备好后按回车键开始 >>> ")
    print()
    print("3 秒后开始，请切换到扫雷窗口...")
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
        print("  3. 使用默认主题（绿色格子）")
        print("  4. 棋盘没有被其他窗口遮挡")
        input("按回车键退出...")
        sys.exit(1)

    board_x, board_y, cell_size = board_info
    print(f"  棋盘位置: ({board_x}, {board_y}), 格子大小: {cell_size}px")

    # 保存调试图片，让用户确认定位是否正确
    save_debug_image(screenshot, board_x, board_y, cell_size)
    print()
    print("请检查 debug_board.png 确认棋盘定位是否正确。")
    resp = input("定位正确吗？(y=继续 / n=退出) >>> ").strip().lower()
    if resp != 'y' and resp != 'yes' and resp != '':
        print("已退出。请调整窗口位置后重试。")
        sys.exit(0)
    print()

    # 内存中跟踪已标旗的格子
    known_flags = set()

    # 第一次点击：棋盘中心（左键点开）
    print("[2] 第一次点击（棋盘中心，左键点开）...")
    center_r, center_c = ROWS // 2, COLS // 2
    click_cell(board_x, board_y, cell_size, center_r, center_c, button='left')
    time.sleep(0.5)

    prev_board = None
    turn = 0
    stale_count = 0
    max_stale = 5

    print("[3] 开始自动求解循环...")
    print()

    while not stopped:
        wait_if_paused()
        if stopped:
            break

        turn += 1
        print(f"--- 第 {turn} 轮 ---")

        # 截图识别
        time.sleep(0.15)
        screenshot = ImageGrab.grab()
        board = read_board(screenshot, board_x, board_y, cell_size, known_flags)

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
            safe, mines_found = constraint_solve(board)

        actions = 0

        # 优先左键点开安全格
        for r, c in safe:
            if stopped:
                break
            wait_if_paused()
            if board[r][c] == UNKNOWN:
                click_cell(board_x, board_y, cell_size, r, c, button='left')
                actions += 1

        # 然后右键标旗
        for r, c in mines_found:
            if stopped:
                break
            wait_if_paused()
            if board[r][c] == UNKNOWN:
                flag_cell(board_x, board_y, cell_size, r, c)
                known_flags.add((r, c))
                board[r][c] = FLAGGED
                actions += 1

        if actions == 0:
            # 没有确定解，概率猜测（左键点开）
            guess = probability_guess(board)
            if guess is None:
                print("  没有可操作的格子，结束。")
                break
            r, c = guess
            print(f"  概率猜测: ({r}, {c})")
            click_cell(board_x, board_y, cell_size, r, c, button='left')
            actions = 1
            stale_count = 0
        else:
            print(f"  执行了 {actions} 个操作 (点开: {len(safe)}, 标雷: {len(mines_found)})")

        # 检测是否卡住
        if prev_board is not None and board == prev_board:
            stale_count += 1
            if stale_count >= max_stale:
                print("  棋盘状态连续无变化，尝试概率猜测...")
                guess = probability_guess(board)
                if guess:
                    r, c = guess
                    print(f"  强制猜测: ({r}, {c})")
                    click_cell(board_x, board_y, cell_size, r, c, button='left')
                stale_count = 0
                time.sleep(0.3)
        else:
            stale_count = 0

        prev_board = [row[:] for row in board]
        time.sleep(0.1)

    print()
    print("程序结束。")
    if HAS_KEYBOARD:
        keyboard.unhook_all()
    if sys.platform == 'win32':
        input("按回车键退出...")


if __name__ == "__main__":
    main()
