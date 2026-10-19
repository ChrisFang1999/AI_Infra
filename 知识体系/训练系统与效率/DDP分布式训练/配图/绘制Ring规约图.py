"""绘制四 rank 经典环形 AllReduce，每轮状态由集合模拟计算，避免手工错标。"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
W, H = 2400, 3100
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
COLORS = {'A':'#2563eb','B':'#8065a9','C':'#628179','D':'#a17949'}
BG='#f5f7fb'; INK='#203049'
im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)

def text(x,y,s,size=30,color=INK,center=False):
    f=ImageFont.truetype(FONT,size)
    b=d.multiline_textbbox((0,0),s,font=f,spacing=10)
    width,height=b[2]-b[0],b[3]-b[1]
    if center:x-=width/2
    assert x>=0 and x+width<W and y+height<H,(s,x,y,width,height)
    d.multiline_text((x,y),s,font=f,fill=color,spacing=10)

def rect(x,y,w,h,color='#ffffff',outline='#d3dce9'):
    d.rounded_rectangle((x,y,x+w,y+h),radius=20,fill=color,outline=outline,width=2)

def arrow(a,b,color,width=5):
    d.line((*a,*b),fill=color,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-18*math.cos(angle-.5),b[1]-18*math.sin(angle-.5)),
                    (b[0]-18*math.cos(angle+.5),b[1]-18*math.sin(angle+.5))],fill=color)

def label(block, contributors=None, gathered=False):
    if gathered:return block+'*'
    return block+'['+','.join(str(r) for r in sorted(contributors))+']'

# 每个 block 用贡献来源集合表示；对应元素的实际值执行 SUM。
state=[{b:{r} for b in 'ABCD'} for r in range(4)]
reduce_steps=[]
for step in range(3):
    sends=[]
    for r in range(4):
        block='ABCD'[(r-1-step)%4]
        sends.append((block,set(state[r][block])))
    received=[]
    for r,(block,parts) in enumerate(sends):
        dst=(r+1)%4
        assert state[dst][block].isdisjoint(parts)
        state[dst][block]|=parts
    for r in range(4):
        block=sends[(r-1)%4][0]
        received.append((block,set(state[r][block])))
    reduce_steps.append((sends,received))
assert all(state[r]['ABCD'[r]]==set(range(4)) for r in range(4))
known=[{ 'ABCD'[r] } for r in range(4)]
gather_steps=[]
for step in range(3):
    sends=['ABCD'[(r-step)%4] for r in range(4)]
    assert all(sends[r] in known[r] for r in range(4))
    for r,b in enumerate(sends):known[(r+1)%4].add(b)
    gather_steps.append((sends,[set(k) for k in known]))
assert all(k==set('ABCD') for k in known)

def ring_panel(x,y,step,phase,sends,results):
    pw,ph=750,930
    rect(x,y,pw,ph)
    text(x+28,y+24,f'第 {step+1} 轮',34)
    text(x+28,y+77,'四条边同时发送；箭头标签是发送内容',25,'#536176')
    pts=[(x+145,y+235),(x+605,y+235),(x+605,y+605),(x+145,y+605)]
    for r in range(4):
        nxt=(r+1)%4;a=pts[r];b=pts[nxt]
        vx,vy=b[0]-a[0],b[1]-a[1];length=math.hypot(vx,vy)
        start=(a[0]+vx/length*62,a[1]+vy/length*62)
        end=(b[0]-vx/length*62,b[1]-vy/length*62)
        block=sends[r][0] if phase=='reduce' else sends[r]
        color=COLORS[block]
        arrow(start,end,color,8 if block=='A' else 4)
        value=label(*sends[r]) if phase=='reduce' else label(block,gathered=True)
        mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
        if r in (0,2):my-=53
        # 白底标签覆盖线段；上下和左右边都能直接看到该轮传递的块。
        f=ImageFont.truetype(FONT,29);bb=d.textbbox((0,0),value,font=f)
        tw=bb[2]-bb[0];rect(mx-tw/2-12,my-4,tw+24,48,'#ffffff','#ffffff')
        text(mx,my,value,29,color,True)
    for r,(cx,cy) in enumerate(pts):
        d.ellipse((cx-60,cy-60,cx+60,cy+60),fill='#e4efff',outline='#b6c4d5',width=2)
        text(cx,cy-17,f'R{r}',34,center=True)
    text(x+375,y+355,'接收后\n与本地块相加' if phase=='reduce' else '接收后保存\n不再累加',29,center=True)
    text(x+28,y+715,'本轮完成后：',27,'#536176')
    for r in range(4):
        xx=x+28+(r%2)*365; yy=y+768+(r//2)*65
        if phase=='reduce':
            block,parts=results[r];value=label(block,parts)
        else:value=' '.join(b+'*' for b in sorted(results[r]))
        text(xx,yy,f'R{r}: {value}',26)

text(60,45,'Ring AllReduce：数据如何沿环规约，再传遍所有 rank',49)
text(60,125,'4 个 rank 的经典单向环：R0 → R1 → R2 → R3 → R0；每轮所有 rank 同时发送、接收。',31,'#536176')
rect(60,205,2280,190,'#eaf0fa')
text(90,230,'初始：每个 rank r 持有完整向量 [A_r, B_r, C_r, D_r]，切成四个等大块。',32)
text(90,285,'记号：A[1,2] = A1 + A2；A* = A0 + A1 + A2 + A3（B*、C*、D* 同理）。',30)
text(90,340,'沿蓝色 A 块看：先边传边累加，得到 A*；再传递完整 A*，不重复相加。',30,COLORS['A'])
text(60,430,'阶段 1  ReduceScatter：每轮传递一个块，接收方累加对应本地贡献',37)
for i,(sends,received) in enumerate(reduce_steps):ring_panel(45+i*785,505,i,'reduce',sends,received)
rect(60,1465,2280,120,'#e4f5ed')
text(90,1490,'3 轮后：R0 持有 A*，R1 持有 B*，R2 持有 C*，R3 持有 D*。',33)
text(90,1540,'每个 rank 获得一块完整归约结果；其他残留局部块不作为此阶段输出。',26,'#536176')
text(60,1635,'阶段 2  AllGather：每轮转发一个已归约的块，逐步收齐所有结果',37)
for i,(sends,known_blocks) in enumerate(gather_steps):ring_panel(45+i*785,1710,i,'gather',sends,known_blocks)
rect(60,2670,2280,160,'#e4efff')
text(90,2695,'最终：每个 rank 都持有 [A*, B*, C*, D*]，SUM AllReduce 完成。',35)
text(90,2755,'没有一个中心 rank 收齐全部原始梯度；各 rank 与环上相邻节点持续交换分块。',29)
text(60,2880,'读图顺序：每个阶段从左到右；R0～R3 的位置和环方向始终不变。',30)
text(60,2940,'这是逻辑环与 SUM 示意；实际 NCCL 可选择其他算法，DDP 的梯度平均还需归一化。',28,'#536176')
im.save(ROOT/'11_Ring_AllReduce分步图.png')
print('六轮贡献来源与最终结果验证通过；图片已生成。')
