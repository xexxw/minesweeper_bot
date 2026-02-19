"""
Minesweeper Bot — Windows 11 Microsoft Minesweeper 自动化
适用于高级模式 30x16, 99 颗雷

使用方式:
  1. 打开扫雷，开始一局高级模式（新游戏，还没点过任何格子）
  2. 运行本程序
  3. 按提示用鼠标点击棋盘左上角格子中心，按 F2 确认
  4. 再点击棋盘右下角格子中心，按 F2 确认
  5. 程序自动开始

快捷键:
  F2  — 确认鼠标位置（定位阶段）
  F9  — 暂停 / 继续
  F10 — 停止退出
"""

import time
import sys
import os
import itertools
from collections import defaultdict

import pyautogui
import numpy as np
from PIL import ImageGrab, ImageDraw

try:
    import keyboard
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
EMPTY = 0

# pyautogui 设置
pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

# 数字颜色 — 运行时从截图学习，不再硬编码
# key=数字, value=RGB tuple
LEARNED_NUMBER_COLORS = {}
NUMBER_TOLERANCE = 55

# ---------------------------------------------------------------------------
# 全局控制
# ---------------------------------------------------------------------------
paused = False
stopped = False
position_confirmed = False
confirmed_pos = (0, 0)

# 运行时从截图采样的颜色（不再硬编码）
UNOPENED_COLORS = []   # 未打开格子的颜色（从左上角采样）
OPENED_COLORS = []     # 已打开格子的颜色（棋盘格有深浅两种）
COLOR_TOLERANCE = 40


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


def on_confirm_position():
    global position_confirmed, confirmed_pos
    confirmed_pos = pyautogui.position()
    position_confirmed = True


def setup_hotkeys():
    if HAS_KEYBOARD:
        keyboard.add_hotkey('F9', on_pause)
        keyboard.add_hotkey('F10', on_stop)
        keyboard.add_hotkey('F2', on_confirm_position)
        print("快捷键: F2=确认位置, F9=暂停/继续, F10=停止")
    else:
        print("错误: 需要 keyboard 库。请运行 pip install keyboard")
        sys.exit(1)


def wait_if_paused():
    while paused and not stopped:
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def color_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1, dtype=float) - np.array(c2, dtype=float)) ** 2))


def is_unopened(pixel):
    """判断像素是否为未打开格子的颜色"""
    for uc in UNOPENED_COLORS:
        if color_distance(pixel, uc) < COLOR_TOLERANCE:
            return True
    return False


def is_opened(pixel):
    """判断像素是否为已打开格子的背景颜色"""
    for oc in OPENED_COLORS:
        if color_distance(pixel, oc) < COLOR_TOLERANCE:
            return True
    return False


def is_red(pixel):
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    return r > 180 and g < 100 and b < 100


# ---------------------------------------------------------------------------
# 手动定位棋盘
# ---------------------------------------------------------------------------
def wait_for_position(prompt):
    """等待用户移动鼠标到目标位置并按 F2 确认"""
    global position_confirmed
    position_confirmed = False
    print(prompt)
    print("  (移动鼠标到目标位置，按 F2 确认)")

    while not position_confirmed and not stopped:
        time.sleep(0.05)

    if stopped:
        return None
    pos = confirmed_pos
    print(f"  已确认: ({pos[0]}, {pos[1]})")
    return pos


def calibrate_board():
    """
    让用户手动点击左上角和右下角格子来定位棋盘。
    返回 (board_x, board_y, cell_size)
    """
    print()
    print("=== 棋盘定位 ===")
    print("请将鼠标移到棋盘【左上角第一个格子的中心】")
    top_left = wait_for_position("  → 左上角格子中心:")
    if top_left is None:
        return None

    print()
    print("请将鼠标移到棋盘【右下角最后一个格子的中心】")
    bottom_right = wait_for_position("  → 右下角格子中心:")
    if bottom_right is None:
        return None

    # 计算格子大小
    # 左上角格子中心到右下角格子中心的距离 = (COLS-1) * cell_size 和 (ROWS-1) * cell_size
    dx = bottom_right[0] - top_left[0]
    dy = bottom_right[1] - top_left[1]

    cell_w = dx / (COLS - 1)
    cell_h = dy / (ROWS - 1)
    cell_size = round((cell_w + cell_h) / 2)

    # 棋盘左上角 = 左上角格子中心 - 半个格子
    board_x = round(top_left[0] - cell_size / 2)
    board_y = round(top_left[1] - cell_size / 2)

    print()
    print(f"  格子大小: {cell_size}px")
    print(f"  棋盘左上角: ({board_x}, {board_y})")

    return board_x, board_y, cell_size


def sample_colors(screenshot, board_x, board_y, cell_size):
    """
    从截图中采样未打开格子的颜色。
    扫雷棋盘格是深浅交替的，采样几个格子取颜色。
    """
    global UNOPENED_COLORS
    img = np.array(screenshot)
    colors = []

    # 采样前几行几列的格子中心颜色
    for r in range(min(3, ROWS)):
        for c in range(min(4, COLS)):
            cx = board_x + c * cell_size + cell_size // 2
            cy = board_y + r * cell_size + cell_size // 2
            if 0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]:
                # 取中心 3x3 区域平均色
                radius = 2
                y1 = max(0, cy - radius)
                y2 = min(img.shape[0], cy + radius + 1)
                x1 = max(0, cx - radius)
                x2 = min(img.shape[1], cx + radius + 1)
                avg = img[y1:y2, x1:x2].mean(axis=(0, 1))
                colors.append(avg)

    if not colors:
        return

    # 聚类：把相近的颜色合并（棋盘格通常有 2 种颜色）
    unique = [colors[0]]
    for c in colors[1:]:
        is_new = True
        for u in unique:
            if color_distance(c, u) < 30:
                is_new = False
                break
        if is_new:
            unique.append(c)

    UNOPENED_COLORS = unique[:4]  # 最多保留 4 种
    print(f"  采样到 {len(UNOPENED_COLORS)} 种未打开格子颜色")
    for i, c in enumerate(UNOPENED_COLORS):
        print(f"    颜色{i+1}: RGB({int(c[0])}, {int(c[1])}, {int(c[2])})")


def save_debug_image(screenshot, board_x, board_y, cell_size):
    """保存调试图片"""
    debug_img = screenshot.copy()
    draw = ImageDraw.Draw(debug_img)

    bw = cell_size * COLS
    bh = cell_size * ROWS
    draw.rectangle([board_x, board_y, board_x + bw, board_y + bh], outline='red', width=3)

    for r in range(ROWS + 1):
        y = board_y + r * cell_size
        draw.line([(board_x, y), (board_x + bw, y)], fill='yellow', width=1)
    for c in range(COLS + 1):
        x = board_x + c * cell_size
        draw.line([(x, board_y), (x, board_y + bh)], fill='yellow', width=1)

    debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_board.png")
    debug_img.save(debug_path)
    print(f"  调试图片: {debug_path}")
    return debug_path


# ---------------------------------------------------------------------------
# 格子识别
# ---------------------------------------------------------------------------
def get_cell_center(board_x, board_y, cell_size, row, col):
    cx = board_x + col * cell_size + cell_size // 2
    cy = board_y + row * cell_size + cell_size // 2
    return cx, cy


def add_opened_color(color):
    """动态添加已打开格子的背景颜色"""
    for oc in OPENED_COLORS:
        if color_distance(color, oc) < 30:
            return
    OPENED_COLORS.append(np.array(color, dtype=float))


def get_cell_region(img_array, board_x, board_y, cell_size, row, col):
    """获取格子的像素区域"""
    x1 = board_x + col * cell_size
    y1 = board_y + row * cell_size
    x2 = x1 + cell_size
    y2 = y1 + cell_size
    # 内缩 2px 避免边界
    margin = 2
    x1 = max(0, x1 + margin)
    y1 = max(0, y1 + margin)
    x2 = min(img_array.shape[1], x2 - margin)
    y2 = min(img_array.shape[0], y2 - margin)
    return img_array[y1:y2, x1:x2]


def identify_cell(img_array, board_x, board_y, cell_size, row, col):
    """识别单个格子状态 — 使用边缘采样 + 动态颜色学习"""
    region = get_cell_region(img_array, board_x, board_y, cell_size, row, col)
    if region.size == 0:
        return UNKNOWN

    h, w = region.shape[:2]

    # 采样边缘像素（上下左右各取一行/列的中间段），避开中心装饰图标
    edge_pixels = []
    mid_h, mid_w = h // 2, w // 2
    # 上边缘
    for x in range(w // 4, 3 * w // 4):
        edge_pixels.append(region[1, x])
    # 下边缘
    for x in range(w // 4, 3 * w // 4):
        edge_pixels.append(region[h - 2, x])
    # 左边缘
    for y in range(h // 4, 3 * h // 4):
        edge_pixels.append(region[y, 1])
    # 右边缘
    for y in range(h // 4, 3 * h // 4):
        edge_pixels.append(region[y, w - 2])

    if not edge_pixels:
        return UNKNOWN

    edge_arr = np.array(edge_pixels, dtype=float)
    edge_avg = edge_arr.mean(axis=0)

    # 1. 判断是否为未打开格子（边缘颜色匹配未打开颜色）
    unopened_match = 0
    for ep in edge_pixels:
        if is_unopened(ep):
            unopened_match += 1
    unopened_ratio = unopened_match / len(edge_pixels)

    if unopened_ratio > 0.6:
        return UNKNOWN

    # 2. 已打开格子 — 判断是空白还是数字
    # 取格子中心区域找非背景色像素（数字文本）
    inner_margin = max(3, cell_size // 5)
    iy1 = inner_margin
    iy2 = h - inner_margin
    ix1 = inner_margin
    ix2 = w - inner_margin
    if iy2 <= iy1 or ix2 <= ix1:
        add_opened_color(edge_avg)
        return EMPTY

    inner = region[iy1:iy2, ix1:ix2]

    # 动态学习已打开背景色
    add_opened_color(edge_avg)

    # 找非背景色像素（即数字文本的像素）
    fg_pixels = []
    for py in range(inner.shape[0]):
        for px in range(inner.shape[1]):
            p = inner[py, px]
            if not is_opened(p) and not is_unopened(p):
                fg_pixels.append(p)

    # 如果前景像素太少，认为是空白格
    fg_ratio = len(fg_pixels) / max(1, inner.shape[0] * inner.shape[1])
    if fg_ratio < 0.05:
        return EMPTY

    # 3. 有前景像素 → 尝试匹配已学习的数字颜色
    fg_arr = np.array(fg_pixels, dtype=float)
    fg_avg = fg_arr.mean(axis=0)

    if LEARNED_NUMBER_COLORS:
        best_num = None
        best_dist = NUMBER_TOLERANCE
        for num, nc in LEARNED_NUMBER_COLORS.items():
            d = color_distance(fg_avg, nc)
            if d < best_dist:
                best_dist = d
                best_num = num
        if best_num is not None:
            return best_num

    # 4. 未学习过的颜色 → 返回特殊标记让 learn_numbers 处理
    # 暂时返回 EMPTY（后续 learn_numbers 会修正）
    return EMPTY


def learn_number_colors(img_array, board_x, board_y, cell_size):
    """
    第一次点击后，扫描已打开区域，学习数字颜色。
    通过聚类前景色来区分不同数字。
    """
    fg_clusters = []  # [(avg_color, count, positions)]

    for r in range(ROWS):
        for c in range(COLS):
            region = get_cell_region(img_array, board_x, board_y, cell_size, r, c)
            if region.size == 0:
                continue
            h, w = region.shape[:2]

            # 检查边缘是否为未打开
            edge_pixels = []
            for x in range(w // 4, 3 * w // 4):
                if 1 < h - 2:
                    edge_pixels.append(region[1, x])
                    edge_pixels.append(region[h - 2, x])
            if not edge_pixels:
                continue
            unopened_count = sum(1 for ep in edge_pixels if is_unopened(ep))
            if unopened_count / len(edge_pixels) > 0.6:
                continue

            # 已打开格子 — 提取前景色
            edge_avg = np.array(edge_pixels, dtype=float).mean(axis=0)
            add_opened_color(edge_avg)

            inner_margin = max(3, cell_size // 5)
            iy1, iy2 = inner_margin, h - inner_margin
            ix1, ix2 = inner_margin, w - inner_margin
            if iy2 <= iy1 or ix2 <= ix1:
                continue

            inner = region[iy1:iy2, ix1:ix2]
            fg_pixels = []
            for py in range(inner.shape[0]):
                for px in range(inner.shape[1]):
                    p = inner[py, px]
                    if not is_opened(p) and not is_unopened(p):
                        fg_pixels.append(p)

            fg_ratio = len(fg_pixels) / max(1, inner.shape[0] * inner.shape[1])
            if fg_ratio < 0.05:
                continue

            fg_avg = np.mean(fg_pixels, axis=0)

            # 聚类到已有簇
            merged = False
            for cluster in fg_clusters:
                if color_distance(fg_avg, cluster[0]) < 40:
                    # 更新平均色
                    n = cluster[1]
                    cluster[0] = (cluster[0] * n + fg_avg) / (n + 1)
                    cluster[1] = n + 1
                    cluster[2].append((r, c))
                    merged = True
                    break
            if not merged:
                fg_clusters.append([fg_avg.copy(), 1, [(r, c)]])

    # 按出现次数排序（数字1最常见，2次之...）
    fg_clusters.sort(key=lambda x: -x[1])

    # 分配数字：出现最多的是1，次多的是2...
    for i, cluster in enumerate(fg_clusters):
        num = i + 1
        if num > 8:
            break
        LEARNED_NUMBER_COLORS[num] = tuple(int(v) for v in cluster[0])
        print(f"    数字 {num}: RGB{LEARNED_NUMBER_COLORS[num]} (出现 {cluster[1]} 次)")

    print(f"  共学习到 {len(LEARNED_NUMBER_COLORS)} 种数字颜色")


def read_board(screenshot, board_x, board_y, cell_size, known_flags=None):
    """读取整个棋盘"""
    img = np.array(screenshot)
    board = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            val = identify_cell(img, board_x, board_y, cell_size, r, c)
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
    """基础规则求解"""
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
    """约束推理求解"""
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
    """概率猜测"""
    unknown_cells = []
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == UNKNOWN:
                unknown_cells.append((r, c))

    if not unknown_cells:
        return None

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
    for cell in unknown_cells:
        if cell in constraint_count and constraint_count[cell] > 0:
            avg_danger[cell] = danger[cell] / constraint_count[cell]
        else:
            total_flags = sum(1 for r in range(ROWS) for c in range(COLS) if board[r][c] == FLAGGED)
            remaining_mines = MINES - total_flags
            remaining_unknown = len(unknown_cells)
            avg_danger[cell] = remaining_mines / max(remaining_unknown, 1)

    def priority(cell):
        r, c = cell
        d = avg_danger.get(cell, 0.5)
        is_corner = (r in (0, ROWS - 1)) and (c in (0, COLS - 1))
        is_edge = r in (0, ROWS - 1) or c in (0, COLS - 1)
        if is_corner:
            d -= 0.05
        elif is_edge:
            d -= 0.02
        return d

    return min(unknown_cells, key=priority)


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
        changes = sum(1 for r in range(ROWS) for c in range(COLS) if board[r][c] != prev_board[r][c])
        if changes > ROWS * COLS * 0.3:
            return True, "LOSE"

    return False, None


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    global stopped

    print("=" * 50)
    print("  扫雷自动化机器人 Minesweeper Bot")
    print("  高级模式 30x16, 99 颗雷")
    print("=" * 50)
    print()

    setup_hotkeys()
    print()
    print("请先打开扫雷，开始一局高级模式新游戏（不要点任何格子）。")
    print()
    input("准备好后按回车键开始定位 >>> ")

    # 手动定位棋盘
    board_info = calibrate_board()
    if board_info is None:
        print("已取消。")
        sys.exit(0)

    board_x, board_y, cell_size = board_info

    # 截图采样颜色
    print()
    print("[1] 采样格子颜色...")
    screenshot = ImageGrab.grab()
    sample_colors(screenshot, board_x, board_y, cell_size)

    # 保存调试图
    save_debug_image(screenshot, board_x, board_y, cell_size)
    print()
    resp = input("请查看 debug_board.png 确认网格对齐。继续? (y/n) >>> ").strip().lower()
    if resp == 'n':
        print("已退出。")
        sys.exit(0)

    # 内存中跟踪已标旗的格子
    known_flags = set()

    # 第一次点击：棋盘中心
    print()
    print("[2] 第一次点击（棋盘中心）...")
    center_r, center_c = ROWS // 2, COLS // 2
    click_cell(board_x, board_y, cell_size, center_r, center_c, button='left')
    time.sleep(0.8)

    # 点击后重新截图，学习已打开格子的颜色和数字颜色
    print()
    print("[2.5] 学习颜色...")
    time.sleep(0.3)
    screenshot2 = ImageGrab.grab()
    img2 = np.array(screenshot2)
    learn_number_colors(img2, board_x, board_y, cell_size)
    save_debug_image(screenshot2, board_x, board_y, cell_size)

    prev_board = None
    turn = 0
    stale_count = 0
    max_stale = 5

    print()
    print("[3] 开始自动求解...")
    print()

    while not stopped:
        wait_if_paused()
        if stopped:
            break

        turn += 1
        print(f"--- 第 {turn} 轮 ---")

        time.sleep(0.15)
        screenshot = ImageGrab.grab()
        board = read_board(screenshot, board_x, board_y, cell_size, known_flags)

        unknown, flagged, opened = count_states(board)
        print(f"  未知: {unknown}, 旗帜: {flagged}, 已开: {opened}")

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

        # 然后标旗
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

        if prev_board is not None and board == prev_board:
            stale_count += 1
            if stale_count >= max_stale:
                print("  连续无变化，强制猜测...")
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
