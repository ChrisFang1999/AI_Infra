from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
S=2
W,H=1600,1060
im=Image.new('RGB',(W*S,H*S),'#f5f7fb');d=ImageDraw.Draw(im)
font='/System/Library/Fonts/STHeiti Medium.ttc'
def f(sz): return ImageFont.truetype(font,sz*S)
def text(x,y,t,size=25,color='#24364b'):
 d.text((x*S,y*S),t,font=f(size),fill=color)
def rect(box,fill,outline=None,r=14,width=1):
 d.rounded_rectangle(tuple(int(v*S) for v in box),radius=r*S,fill=fill,outline=outline,width=width*S)
def line(pts,fill='#90a3b7',width=2):d.line([(int(x*S),int(y*S)) for x,y in pts],fill=fill,width=width*S)
def arrow(x1,y1,x2,y2,color='#8296ac'):
 line([(x1,y1),(x2,y2)],color,3)
 d.polygon([(x2*S,y2*S),((x2-11)*S,(y2-7)*S),((x2-11)*S,(y2+7)*S)],fill=color)
text(64,42,'为什么通信等待时 GPU 利用率仍是 100%？',42,'#152c47')
text(64,105,'关键：利用率反映 kernel 执行的时间占比，不直接反映训练进度。',25,'#586d84')
rect((64,162,1536,634),'#ffffff')
text(92,187,'典型场景：一个 rank 未推进，其他 rank 已进入通信等待',28)
text(460,240,'以下完整区间为一个利用率统计窗口',22,'#637891')
arrow(460,279,1450,279)
text(1470,263,'时间',20,'#637891')
rows=[(308,'rank 0','CPU 数据等待','未提交对应通信','#e5e9ef','#566779'),(384,'rank 1','NCCL kernel：等待 / 轮询','仍有 kernel 执行','#dceaff','#245da1'),(460,'rank 2','NCCL kernel：等待 / 轮询','仍有 kernel 执行','#dceaff','#245da1'),(536,'rank 3','NCCL kernel：等待 / 轮询','仍有 kernel 执行','#dceaff','#245da1')]
for y,r,l,extra,bg,fg in rows:
 text(95,y+12,r,25)
 text(210,y+15,extra,20,'#62738a')
 rect((460,y,1450,y+54),bg,r=8)
 text(488,y+13,l,24,fg)
 if r!='rank 0':
  for x in range(990,1420,84):
   d.arc(((x+2)*S,(y+14)*S,(x+24)*S,(y+36)*S),start=30,end=330,fill=fg,width=2*S)
   d.polygon([((x+24)*S,(y+17)*S),((x+15)*S,(y+17)*S),((x+23)*S,(y+25)*S)],fill=fg)
text(65,660,'等待 rank 的指标与训练表现',28)
rect((64,711,530,894),'#e8f1ff',r=16)
rect((552,711,1018,894),'#edf4f2',r=16)
rect((1040,711,1536,894),'#fff0e6',r=16)
text(88,735,'GPU 利用率',24,'#456384')
text(88,779,'100%',48,'#215ba3')
text(88,844,'整个窗口都有通信 kernel 执行',21,'#456384')
text(576,735,'GPU 功率',24,'#4c7067')
text(576,779,'可能较低',39,'#2e695b')
text(576,844,'等待活动弱于密集训练计算',21,'#4c7067')
text(1064,735,'训练进度',24,'#88684d')
text(1064,779,'step 不推进',39,'#a7632f')
text(1064,844,'通信未完成，后续训练受阻',21,'#88684d')
text(64,923,'持续执行 kernel  ≠  持续进行有效训练计算',31,'#152c47')
text(64,980,'示意仅展示通信等待路径；统计窗口是指标自身口径，不是监控读取间隔。',21,'#637891')
text(64,1016,'GPU 上仍在执行等待 / 轮询指令；100% 利用率不能单独证明发生 Hang。',21,'#637891')
p=Path('/private/tmp/GPU通信等待利用率原理示意图.png');im.save(p)
print(p)
