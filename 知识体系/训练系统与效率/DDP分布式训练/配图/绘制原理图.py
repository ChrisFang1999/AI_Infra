"""可重复生成教程 PNG；Pillow 绘制，坐标与文本边界自动检查。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
ROOT = Path(__file__).parent
FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
W = 1800
COLORS = ['#e4efff', '#e4f5ed', '#fff1d9', '#eee9fb']

def new(title, subtitle, h=1000):
    global im, d
    im = Image.new('RGB', (W,h), '#f5f7fb'); d=ImageDraw.Draw(im)
    text(65,40,title,46); text(65,110,subtitle,25,'#536176')

def text(x,y,s,size=28,color='#203049'):
    font=ImageFont.truetype(FONT,size)
    box=d.multiline_textbbox((x,y),s,font=font,spacing=13)
    assert box[2]<W-20 and box[3]<im.height-10, (s,box)
    d.multiline_text((x,y),s,font=font,fill=color,spacing=13)

def box(x,y,w,h,s,c=0,size=28):
    d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=COLORS[c],outline='#b6c4d5',width=2)
    f=ImageFont.truetype(FONT,size)
    b=d.multiline_textbbox((0,0),s,font=f,spacing=13)
    assert b[2]-b[0] <= w-30 and b[3]-b[1]<=h-20,(s,w,h,b)
    text(x+20,y+20,s,size)

def arrow(x,y,xx,yy):
    d.line((x,y,xx,yy),fill='#62728a',width=4)
    a=math.atan2(yy-y,xx-x)
    p=[(xx,yy),(xx-15*math.cos(a-.5),yy-15*math.sin(a-.5)),(xx-15*math.cos(a+.5),yy-15*math.sin(a+.5))]
    d.polygon(p,fill='#62728a')

def save(name,note='结构与时间仅作原理示意，不代表实测耗时。'):
    text(65,im.height-65,note,23,'#536176'); im.save(ROOT/(name+'.png'))

new('01  DDP 训练全流程','两机各 8 卡：启动阶段一次，训练迭代重复执行',920)
steps=['torchrun 启动\n每机 8 个进程','绑定 GPU\n初始化进程组','CUDA AllReduce\n通信验证','构建数据链路\n创建 DDP 模型','取本地 batch\n搬运到 GPU','本地 forward\n计算 loss','backward\n梯度同步','optimizer\n本地参数更新']
for i,s in enumerate(steps):
    row=i//4; col=i%4;x=65+col*430;y=220+row*300
    box(x,y,380,140,s,i%4)
    if col<3:arrow(x+380,y+70,x+425,y+70)
arrow(1545,360,1545,450);arrow(1545,450,245,450);arrow(245,450,245,520)
text(65,735,'梯度累计：多个本地前向 / 反向，可对应一次同步与参数更新。',28)
save('01_DDP训练全流程图')

new('02  单机与双机进程结构','单机：WORLD_SIZE=8；双机：WORLD_SIZE=16（固定节点示例）',1050)
for node in range(2):
    x=65+node*860
    box(x,210,810,675,f'机器 {node}：本机 torchrun / agent',node,30)
    for local in range(8):
        col=local%2;row=local//2
        box(x+25+col*385,295+row*140,365,115,f'rank {node*8+local}  →  GPU {local}\nLOCAL_RANK={local}',node,25)
text(65,920,'每台机器分别运行启动命令；rank 是进程身份，GPU 编号是本地可见设备编号。',26)
save('02_单机与双机进程结构图')

new('03  初始化与通信验证','协调成员、创建通信组、验证数据通信是不同职责',1000)
for i,(s,c) in enumerate([('训练进程读取环境变量\nRANK / WORLD_SIZE / LOCAL_RANK',0),('绑定本地 GPU → init_process_group\n默认组中各 rank 进入一致的初始化流程',1),('CUDA Tensor AllReduce：每个 rank 输入 1\n16 个 rank 都得到 16，验证这次 collective',2),('创建模型 → DDP 包装\n默认初始化同步模型参数与 buffer',3)]):
    box(65,205+i*170,1100,130,s,c)
    if i<3:arrow(615,335+i*170,615,370+i*170)
box(1220,230,515,520,'Rendezvous：成员协调\n\nStore：交换初始化信息\n\nProcessGroup：通信参与组\n\nNCCL：GPU 通信后端',0,27)
save('03_分布式初始化与通信验证图','通信资源可延迟创建；小 Tensor 成功不代表带宽或全部链路已验证。')

new('04  模型状态与同步范围','本图针对标准 DDP 完整副本；显式关闭同步及自定义逻辑需另行分析',920)
rows=[('参数 θ','默认构造同步','各 rank 使用同步梯度分别更新'),('buffer','默认初始化同步','启用运行时同步时在 forward 前广播'),('梯度 .grad','backward 中同步','no_sync 可延后至累计窗口末尾'),('优化器状态','DDP 不负责同步','各 rank 一致初始化 / 恢复、执行更新')]
for i,(a,b,c) in enumerate(rows):
    y=220+i*140
    box(65,y,300,110,a,i);box(395,y,480,110,b,i);box(905,y,830,110,c,i,26)
save('04_模型状态与同步范围图','普通 BatchNorm 使用本地统计；SyncBatchNorm 会增加前向通信。')

new('05  全局索引如何分给 16 个 rank','以 64 个样本、shuffle=False 展示；开启 shuffle 后对共同排列做同样划分',900)
box(65,210,1670,100,'相同的全局索引顺序：0、1、2、…、15、16、17、…、63',0)
for i,r in enumerate([0,1,2,15]):
    x=65+i*430;arrow(900,310,x+190,430)
    box(x,430,380,190,f'rank {r}\n{r}、{r+16}、{r+32}、{r+48}',i)
text(65,700,'各 rank 独立计算自己的索引，不由 rank 0 广播样本内容。',29)
save('05_样本索引分配图','不能整除时，Sampler 可补齐重复索引或丢弃尾部，取决于 drop_last。')

new('06  一个 rank 的数据流水线','Map-style Dataset + 自动组 batch + num_workers > 0',1100)
box(65,200,1670,130,'训练进程（CPU）：DistributedSampler → BatchSampler → 分发一批样本索引',0)
for i in range(3):
    x=65+i*570;arrow(x+260,330,x+260,425)
    box(x,425,530,200,f'worker {i}（CPU）\nDataset.__getitem__ × 多个样本\n读取 / 处理 → collate_fn',1,27)
    arrow(x+260,625,x+260,710)
box(65,710,1670,120,'结果队列 / 可选 pin memory → 训练进程 next(iterator) 获取 batch',2)
arrow(900,830,900,870)
box(65,875,1670,115,'训练进程：CPU → GPU 数据搬运 → 本地 forward / backward',3)
save('06_单rank数据加载链路图','默认一个 worker 完成一个 batch 任务；num_workers=0 时读取与 collate 在训练进程中执行。')

new('07  数据预取与消费','预取提供缓冲；等待下一个按序结果不等于所有 worker 都没有完成任务',1000)
for row,(title,ready,need) in enumerate([('供给充足：batch 提前就绪',580,850),('短时波动：缓冲仍可覆盖',800,850),('持续不足：训练开始等待',1300,850)]):
    y=230+row*220;text(65,y,title,29)
    box(400,y+55,ready-400,65,'准备数据',1,23)
    arrow(400,y+155,1640,y+155)
    d.line((need,y+40,need,y+165),fill='#536176',width=2)
    text(need-30,y+170,'取数开始',23)
    if ready>need:box(need,y+55,ready-need,65,'data_get_time：等待结果',2,23)
text(65,890,'取数计时不等于样本处理总耗时；搬运时间应单独定义。',27)
save('07_数据预取与消费时间线')

new('08  独立计算，同步梯度，分别更新','局部仅画 2 个 rank；θ 相同，数据不同，本地梯度一般不同',950)
for i in range(2):
    y=230+i*270
    for j,s in enumerate([f'rank {i}\n参数 θ',f'本地 batch {i}\nforward / loss',f'backward\n本地梯度 g{i}', '同步后梯度\n(g0 + g1) / 2']):
        x=65+j*430;box(x,y,380,130,s,j)
        if j<3:arrow(x+380,y+65,x+425,y+65)
text(65,770,'各 rank：相同 θ + 相同平均梯度 + 相同优化器状态与更新规则 → 一致的新参数',28)
save('08_各rank计算与参数更新图')

new('09  bucket 与梯度就绪','bucket 可包含多个参数梯度，不要求与网络层一一对应',900)
box(65,210,1670,100,'反向传播逐步产生梯度 → autograd hook 标记梯度就绪',0)
for i,s in enumerate(['bucket A\n参数 p1：就绪\n参数 p2：就绪\n可按一致顺序提交通信','bucket B\n参数 p3：就绪\n参数 p4：未就绪\n继续等待本地梯度','bucket C\n参数 p5：未就绪\n参数 p6：未就绪\n后续反向计算继续']):
    box(65+i*570,410,530,270,s,i+1,29)
save('09_bucket划分与梯度就绪图','各 rank 必须遵循一致的 collective 顺序；本地就绪不等于全局通信完成。')

new('10  计算与通信重叠','同一时间轴；色块长度为示意。更早提交不保证总体更快。',1100)
for i,title in enumerate(['较大 bucket：较晚开始通信','多个 bucket：争取覆盖通信']):
    y=230+i*370;text(65,y,title,30)
    text(65,y+90,'反向计算',26);text(65,y+195,'通信',26)
    for j in range(3):box(330+j*300,y+65,300,80,'梯度就绪 '+chr(65+j),0,25)
    if i==0:box(1230,y+170,450,85,'AllReduce → 完成',2,25)
    else:
        for j in range(3):box(630+j*300,y+170,285,85,'提交 '+chr(65+j)+' → 完成',1,23)
    arrow(330,y+290,1700,y+290)
text(65,990,'小 bucket：调用更多；大 bucket：等待更久。最后通信可能留下尾部等待。',27)
save('10_计算通信重叠时间线')

# Ring 图独立生成，集合模拟验证每轮贡献，避免手工错标。
import runpy
runpy.run_path(str(ROOT / '绘制Ring规约图.py'))

new('12  梯度累计与 no_sync','示例：每 rank micro-batch=4，16 rank，累计 4 次 → 有效全局 batch=256',1000)
for i in range(4):
    x=65+i*430
    box(x,230,380,350,f'micro-step {i+1}\n\n取数据 → forward\nbackward\n\n'+('no_sync：本地累计' if i<3 else '同步全部累计贡献'),i,27)
    if i<3:arrow(x+380,400,x+425,400)
box(65,710,1670,120,'一次参数更新：梯度处理 → optimizer.step → 清理梯度',3)
save('12_梯度累计与no_sync时间线','no_sync 同时覆盖 forward / backward；最后同步仍可包含多个 bucket 通信。')

new('13  AMP 与参数更新顺序','FP16 + GradScaler；每个累计窗口统一进行 unscale 和 scaler.update',1150)
items=['每个 micro-batch：autocast forward / loss → loss 权重归一化', '同一窗口使用相同 scale：scaled backward → 最后同步累计梯度', '窗口末尾：unscale → 梯度裁剪 → scaler.step 检查 Inf / NaN']
for i,s in enumerate(items):
    box(65,210+i*160,1670,115,s,i,28)
    arrow(900,325+i*160,900,365+i*160)
box(65,720,790,160,'梯度有限\n执行 optimizer.step\n推进按成功更新次数计的 scheduler',1,27)
box(945,720,790,160,'梯度非有限\n跳过 optimizer.step\n不推进上述 scheduler',2,27)
box(65,950,1670,100,'两分支均：scaler.update → 清理梯度 → 下一累计窗口',3)
save('13_AMP与参数更新顺序图')

new('14  迟到等待与 collective 不匹配','通信持续时间较长，不直接等于网络传输慢',1100)
text(65,195,'情况 A：同一次 collective，有 rank 迟到',32)
box(65,270,350,95,'rank 0～14',0);box(450,270,500,95,'本地计算 → 提交通信 A',0);box(990,270,745,95,'等待 rank 15 → 通信完成',2)
box(65,405,350,95,'rank 15',1);box(450,405,800,95,'取数据较慢 → 本地计算 → 提交通信 A',1);box(1290,405,445,95,'共同完成',1)
text(65,625,'情况 B：调用序列或元信息不匹配',32)
box(65,710,790,170,'rank 0：collective A\n例如 AllReduce，元素数为 M\n一直等待或报错',0)
box(945,710,790,170,'rank 15：跳过 A / 提交不匹配操作\n不能按预期共同完成 A\n应检查代码与对应调用信息',2,27)
save('14_慢rank与通信不匹配对照图','本图展示两类机制，不覆盖网络故障、GPU 异常等全部原因。')
print('已生成',len(list(ROOT.glob('*.png'))),'张图')
