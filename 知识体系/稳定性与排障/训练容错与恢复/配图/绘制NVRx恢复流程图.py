from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).parent
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
INK, BLUE, GREEN, RED, GRAY = '#172b4d', '#2463ba', '#16816b', '#bc4949', '#617289'

def label(draw, xy, text, size=31, color=INK):
    draw.multiline_text(xy, text, font=ImageFont.truetype(FONT, size), fill=color, spacing=16)

def box(draw, rect, title, text, color=BLUE):
    draw.rounded_rectangle(rect, radius=20, fill='white', outline=color, width=3)
    x, y, _, _ = rect
    label(draw, (x+25, y+22), title, 36, color)
    label(draw, (x+25, y+90), text, 29)

def arrow(draw, points, color=BLUE):
    draw.line(points, fill=color, width=5, joint='curve')
    a, b = points[-2:]
    ang = math.atan2(b[1]-a[1], b[0]-a[0])
    draw.polygon([b, (b[0]-20*math.cos(ang-.45), b[1]-20*math.sin(ang-.45)),
                  (b[0]-20*math.cos(ang+.45), b[1]-20*math.sin(ang+.45))], fill=color)

def main():
    im = Image.new('RGB', (1800, 1130), '#f5f8fc')
    draw = ImageDraw.Draw(im)
    label(draw, (70, 40), 'NVRx：发现训练停滞，再从有效状态恢复', 48)
    label(draw, (70, 115), 'v0.7.0 作业内重启主线 · ft_launcher + 进度监控 + 应用 resume + 外部资源协同', 30, GRAY)
    items = [
        (80, 215, '1 监测训练进展', '各 rank 上报 heartbeat\n或进入 / 退出受监控 section', BLUE),
        (675, 215, '2 发现超时或退出', 'rank monitor 检测缺失更新\nlauncher 也可发现 rank 退出', RED),
        (1270, 215, '3 结束旧 worker 组', '停止旧训练与通信\n按重试预算进入恢复流程', BLUE),
        (80, 610, '4 检查资源、重新组网', '按配置进行健康检查\n可用资源参与 rendezvous', BLUE),
        (675, 610, '5 启动并恢复状态', '重新初始化训练进程\n应用加载完整 checkpoint', BLUE),
        (1270, 610, '6 验证有效训练', '确认实际 update 继续\n检查吞吐与是否再次失败', GREEN),
    ]
    for x, y, title, text, color in items:
        box(draw, (x, y, x+450, y+225), title, text, color)
    for y in (325, 720):
        arrow(draw, [(530, y), (675, y)])
        arrow(draw, [(1125, y), (1270, y)])
    arrow(draw, [(1495, 440), (1495, 515), (305, 515), (305, 610)])
    draw.rounded_rectangle((80, 900, 1720, 1060), radius=18, fill='#e8eff9')
    label(draw, (110, 925), '恢复所需条件：有效 checkpoint、正确的 resume、可用资源，以及明确的重试上限。', 31)
    label(draw, (110, 985), '应用负责状态接续；平台负责节点替换。无进展或不可恢复时，应按策略停止并保留证据。', 30, GRAY)
    im.save(OUT/'NVRx训练故障检测与恢复流程图.png')

if __name__ == '__main__':
    main()
