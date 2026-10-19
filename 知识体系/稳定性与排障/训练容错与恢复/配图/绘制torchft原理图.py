from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).parent
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
INK, BLUE, GREEN, RED, GRAY = '#172b4d', '#2463ba', '#16816b', '#bc4949', '#617289'

def canvas(height, title, subtitle):
    im = Image.new('RGB', (1800, height), '#f5f8fc')
    d = ImageDraw.Draw(im)
    label(d, (70, 40), title, 48)
    label(d, (70, 110), subtitle, 29, GRAY)
    return im, d

def label(d, xy, text, size=32, color=INK):
    d.multiline_text(xy, text, font=ImageFont.truetype(FONT, size), fill=color, spacing=14)

def box(d, rect, title, lines, color=BLUE, fill='white'):
    d.rounded_rectangle(rect, radius=20, fill=fill, outline=color, width=3)
    x, y, _, _ = rect
    label(d, (x+24, y+20), title, 34, color)
    label(d, (x+24, y+78), lines, 29)

def arrow(d, points, color=BLUE, width=5):
    d.line(points, fill=color, width=width, joint='curve')
    a, b = points[-2:]
    ang = math.atan2(b[1]-a[1], b[0]-a[0])
    d.polygon([b, (b[0]-18*math.cos(ang-.45), b[1]-18*math.sin(ang-.45)),
               (b[0]-18*math.cos(ang+.45), b[1]-18*math.sin(ang+.45))], fill=color)

def draw_logic():
    im, d = canvas(1420, 'torchft：一个副本故障后，训练如何继续', '教学示意：A、B、C 各持有一份逻辑模型；能否继续仍取决于 quorum、状态和通信条件')
    box(d, (545, 180, 1255, 340), 'Lighthouse（灯塔）', '协调成员与 quorum；不转发梯度')
    for x, name, color, text in [(100, '副本组 A', GREEN, 'Manager + 训练进程\n健康'), (680, '副本组 B', RED, 'Manager + 训练进程\n故障 / 掉队'), (1260, '副本组 C', GREEN, 'Manager + 训练进程\n健康')]:
        arrow(d, [(900, 340), (900, 380), (x+220, 380), (x+220, 420)])
        box(d, (x, 420, x+440, 610), name, text, color)
    label(d, (100, 645), '控制协调走 Lighthouse；梯度同步和恢复状态传输走副本之间的连接。', 31, GRAY)
    steps = [
        (100, '1 发现失败', '心跳 / 通信 / 协议超时\n将失败交给容错流程', RED),
        (680, '2 控制提交', '异常梯度不得直接更新\n本组遵循提交决定', BLUE),
        (1260, '3 重组后继续', '满足条件时 A、C 继续\n检测与重组可能有等待', GREEN),
    ]
    for x, title, text, color in steps:
        box(d, (x, 735, x+440, 935), title, text, color)
    arrow(d, [(540, 835), (680, 835)])
    arrow(d, [(1120, 835), (1260, 835)])
    box(d, (100, 1020, 830, 1220), '4 平台重建故障副本', '停止旧进程，隔离故障节点\n在可用资源上启动 B 的新运行实例', GRAY)
    box(d, (970, 1020, 1700, 1220), '5 恢复状态，再参与训练', '从健康同伴取一致的较新状态\n完成协调与恢复后，按协议重新参与', GREEN)
    arrow(d, [(1480, 935), (1480, 975), (465, 975), (465, 1020)], GRAY)
    arrow(d, [(830, 1120), (970, 1120)], GREEN)
    label(d, (100, 1285), '底线：心跳正常不等于训练有进展；无有效 quorum / 无可用恢复状态时，不能承诺继续。', 30, GRAY)
    im.save(OUT/'torchft容错逻辑与副本组示意图.png')

def draw_k8s():
    im, d = canvas(1250, 'torchft 在 Kubernetes 中的职责划分', '推荐部署示意，不是 torchft 的强制运行环境；图中一个 Pod 对应一个副本组仅用于说明')
    box(d, (80, 200, 810, 410), '独立协调服务 · CPU 资源', 'Lighthouse：单一活动实例\nDeployment + Service：稳定入口与故障重建', BLUE)
    box(d, (990, 200, 1720, 410), '训练作业控制器 · 平台负责', '保留健康组；停止并重建故障组\n隔离故障节点、分配新资源、限制重试', GRAY)
    for x, name in [(80, 'GPU 节点 A'), (680, 'GPU 节点 B（故障）'), (1280, 'GPU 节点 C')]:
        color = RED if x == 680 else GREEN
        box(d, (x, 540, x+440, 760), name, '训练 Pod / 副本组\nManager + 8 个 GPU rank', color)
        arrow(d, [(445, 410), (445, 470), (x+220, 470), (x+220, 540)])
    label(d, (80, 795), '蓝线：所有组连接同一 Lighthouse；副本间另需开放通信、Store 和状态传输连接。', 30, GRAY)
    box(d, (80, 900, 810, 1100), '新 GPU 节点 D', '平台启动故障组的新实例\n连接同一 Lighthouse，向健康组恢复状态', GREEN)
    box(d, (990, 900, 1720, 1100), '外部持久存储与观测', '周期性 checkpoint + 日志 / 指标 / 堆栈\n覆盖全部副本丢失与历史现场保留', GRAY)
    arrow(d, [(1355, 410), (1760, 410), (1760, 855), (445, 855), (445, 900)], GRAY)
    label(d, (80, 1160), 'Service 不保存协调状态；单实例自动重建不等于高可用。节点维修仍由平台 / 运维负责。', 30, GRAY)
    im.save(OUT/'torchft在Kubernetes中的部署与恢复示意图.png')

if __name__ == '__main__':
    draw_logic()
    draw_k8s()
