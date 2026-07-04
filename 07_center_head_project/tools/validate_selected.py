import argparse, json
from pathlib import Path
import numpy as np

NAMES=[("reg",2),("height",1),("dim",3),("rot",2),("hm",3)]
def load(path,shape): return np.fromfile(path,np.float32).reshape(shape)
def conv_vector(x,w,b,y,xp):
    patch=np.zeros((x.shape[0],3,3),np.float32)
    y0=max(0,y-1);y1=min(x.shape[1],y+2);x0=max(0,xp-1);x1=min(x.shape[2],xp+2)
    patch[:,y0-y+1:y1-y+1,x0-xp+1:x1-xp+1]=x[:,y0:y1,x0:x1]
    return np.asarray(np.tensordot(w.astype(np.float64),patch.astype(np.float64),axes=((1,2,3),(0,1,2)))+b,dtype=np.float32)
def main():
    p=argparse.ArgumentParser();p.add_argument("--rpn-dir",type=Path,required=True);p.add_argument("--weight-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    rpn=load(a.rpn_dir/"rpn_features.bin",(384,468,468)); eps=np.float32(1e-3); points=[(0,0),(234,234),(467,467)]; results=[]
    sw=load(a.weight_dir/"shared_weight.bin",(64,384,3,3));sb=load(a.weight_dir/"shared_bias.bin",(64,));bn=[load(a.weight_dir/f"shared_bn_{n}.bin",(64,)) for n in ["weight","bias","mean","var"]]
    # Final branch pixels need a 5x5 RPN neighborhood. Compute only requested receptive fields.
    for name,outc in NAMES:
        hw=load(a.weight_dir/f"{name}_hidden_weight.bin",(64,64,3,3));hb=load(a.weight_dir/f"{name}_hidden_bias.bin",(64,));hbn=[load(a.weight_dir/f"{name}_hidden_bn_{n}.bin",(64,)) for n in ["weight","bias","mean","var"]];ow=load(a.weight_dir/f"{name}_output_weight.bin",(outc,64,3,3));ob=load(a.weight_dir/f"{name}_output_bias.bin",(outc,));actual=load(a.output_dir/f"{name}.bin",(outc,468,468))
        for oc,(y,xp) in [(0,q) for q in points]:
            hidden=np.zeros((64,3,3),np.float32)
            for dy in range(-1,2):
                for dx in range(-1,2):
                    yy,xx=y+dy,xp+dx
                    if not(0<=yy<468 and 0<=xx<468): continue
                    shared=np.empty((64,3,3),np.float32)
                    for sy in range(-1,2):
                        for sx in range(-1,2):
                            y2,x2=yy+sy,xx+sx
                            if 0<=y2<468 and 0<=x2<468:
                                v=conv_vector(rpn,sw,sb,y2,x2);shared[:,sy+1,sx+1]=np.maximum((v-bn[2])/np.sqrt(bn[3]+eps)*bn[0]+bn[1],0)
                            else: shared[:,sy+1,sx+1]=0
                    v=conv_vector(shared,hw,hb,1,1);hidden[:,dy+1,dx+1]=np.maximum((v-hbn[2])/np.sqrt(hbn[3]+eps)*hbn[0]+hbn[1],0)
            expected=conv_vector(hidden,ow,ob,1,1)[oc];got=actual[oc,y,xp];results.append({"branch":name,"channel":oc,"y":y,"x":xp,"expected":float(expected),"actual":float(got),"abs_diff":float(abs(expected-got))})
    maximum=max(x["abs_diff"] for x in results);print(json.dumps({"samples":results,"max_abs_diff":maximum},indent=2));raise SystemExit(0 if maximum<2e-4 else 1)
if __name__=="__main__": main()
