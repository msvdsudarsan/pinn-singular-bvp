import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
fig,ax=plt.subplots(figsize=(6.6,3.7));ax.set_xlim(-1.5,11.5);ax.set_ylim(-1.2,7.2);ax.axis('off')
def box(x,y,w,h,txt,c='#e8eef9'):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc=c,ec='k',lw=1.0));ax.text(x,y,txt,ha='center',va='center',fontsize=7)
def line(*pts):
    xs,ys=zip(*pts);ax.plot(xs,ys,'k',lw=1.0)
def dot(x,y,lab,dx=0.3,dy=0.55,col='k'):
    ax.plot(x,y,'o',color=col,ms=5);ax.text(x+dx,y+dy,lab,fontsize=8.5,fontweight='bold')
def gnd(x,y):
    line((x,y),(x,y-0.2));
    for k,w in enumerate((0.35,0.22,0.1)): line((x-w,y-0.2-0.12*k),(x+w,y-0.2-0.12*k))
# main rail y=3 : nodes 1,2,3
X={'1':1,'2':5,'3':9};Y=3
for n,x in X.items(): dot(x,Y,'node '+n)
line((X['1'],Y),(X['1']+0.5,Y));box(2.2,Y,1.9,0.55,r'$L_1,\,R_{l1}$');line((3.15,Y),(X['2'],Y))
line((X['2'],Y),(X['2']+0.5,Y));box(6.2,Y,1.9,0.55,r'$L_2,\,R_{l2}$');line((7.15,Y),(X['3'],Y))
ax.text(2.2,Y-0.6,'$i_1\\to$',ha='center',fontsize=7);ax.text(6.2,Y-0.6,'$i_2\\to$',ha='center',fontsize=7)
# shunt branches down from each node : capacitor + conductance
for n,x,C,G in (('1',1,'$C_1$','$G_1$'),('2',5,'$C_2$','$G_2$'),('3',9,'$C_3$','$G_3$')):
    xc=x-0.45;xg=x+0.45
    line((x,Y),(x,Y-0.35));line((xc,Y-0.35),(xg,Y-0.35))
    line((xc,Y-0.35),(xc,Y-0.8));line((xc-0.25,Y-0.8),(xc+0.25,Y-0.8));line((xc-0.25,Y-0.95),(xc+0.25,Y-0.95));ax.text(xc-0.1,Y-1.3,C,ha='right',fontsize=7)
    line((xc,Y-0.95),(xc,0.3));gnd(xc,0.3)
    line((xg,Y-0.35),(xg,Y-0.7));box(xg,Y-1.05,0.4,0.7,'',c='#fdf2e0');ax.text(xg+0.3,Y-1.05,G,fontsize=7,va='center');line((xg,Y-1.4),(xg,0.3));gnd(xg,0.3)
# current sources
for x,lab in ((1,'$u_1(t)$'),(9,'$u_3(t)$')):
    ax.add_patch(FancyArrowPatch((x,Y+1.9),(x,Y+0.35),arrowstyle='-|>',mutation_scale=14,color='crimson',lw=2));ax.text(x+(0.15 if x<5 else -0.15),Y+1.5,lab,color='crimson',fontsize=8.5,ha='left' if x<5 else 'right')
# resistive bridge on top: node1 - R1a - a - Rab - b - Rb3 - node3 ; a,b are ALGEBRAIC nodes (no capacitor)
yb=6.3
line((X['1']-0.45,Y),(X['1'],Y));line((X['1']-0.45,Y),(X['1']-0.45,yb));box(0.55+0.0,yb,0.01,0.01,'')
line((X['1']-0.45,yb),(2.15,yb));box(2.8,yb,1.3,0.5,'$R_{1a}$',c='#fdf2e0');line((3.45,yb),(4.1,yb))
dot(4.1,yb,'a',dy=0.3,col='#2a7a2a');box(5.3,yb,1.3,0.5,'$R_{ab}$',c='#fdf2e0');line((4.1,yb),(4.65,yb));line((5.95,yb),(6.7,yb))
dot(6.7,yb,'b',dy=0.3,col='#2a7a2a');box(7.9,yb,1.3,0.5,'$R_{b3}$',c='#fdf2e0');line((6.7,yb),(7.25,yb));line((8.55,yb),(9.45,yb));line((9.45,yb),(9.45,Y));line((9,Y),(9.45,Y))
ax.text(5.4,6.95,'algebraic nodes $a$, $b$ (no capacitor)',ha='center',fontsize=7,color='#2a7a2a')
for x,lab in ((4.1,'$R_{a0}$'),(6.7,'$R_{b0}$')):
    line((x,yb),(x,yb-0.5));box(x,yb-0.95,0.4,0.7,'',c='#fdf2e0');ax.text(x+0.3,yb-0.95,lab,fontsize=7,va='center');line((x,yb-1.3),(x,yb-1.8));gnd(x,yb-1.8)
ax.text(5,-0.9,r'State $x=[v_1,v_2,v_3,i_1,i_2,v_a,v_b]^{\top}$,  $E=\mathrm{diag}(C_1,C_2,C_3,L_1,L_2,0,0)$: 5 differential + 2 algebraic variables (index one)',ha='center',fontsize=7)
plt.tight_layout();import os; plt.savefig(os.path.join(os.environ.get('FIGDIR','.'),'fig_network_topology.pdf'))
