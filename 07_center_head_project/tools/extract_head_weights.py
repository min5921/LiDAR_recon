import argparse, collections, io, json, pickle, zipfile
from pathlib import Path
import numpy as np

class StorageRef:
    def __init__(self, value): self.value=value
def rebuild(storage, offset, size, stride, *unused): return {"storage":storage,"offset":int(offset),"shape":tuple(size),"stride":tuple(stride)}
class Unpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=="collections" and name=="OrderedDict": return collections.OrderedDict
        if module=="torch._utils" and name.startswith("_rebuild_tensor"): return rebuild
        if module.startswith("torch") and name.endswith("Storage"): return type(name,(),{"storage_name":name})
        return super().find_class(module,name)
    def persistent_load(self,value): return StorageRef(value)
def tensor(archive,root,item):
    pid=item["storage"].value
    if pid[1].storage_name!="FloatStorage": raise ValueError("only FloatStorage is supported")
    raw=archive.read(f"{root}/data/{pid[2]}"); shape=tuple(int(x) for x in item["shape"]); stride=tuple(int(x)*4 for x in item["stride"])
    return np.ascontiguousarray(np.ndarray(shape=shape,dtype="<f4",buffer=raw,offset=item["offset"]*4,strides=stride))
def save(archive,root,state,key,path):
    value=tensor(archive,root,state[key]); value.tofile(path); return list(value.shape)
def export_conv(archive,root,state,out,prefix,key,bn):
    info={"prefix":prefix,"checkpoint_prefix":key}
    info["weight_shape"]=save(archive,root,state,key+".weight",out/f"{prefix}_weight.bin")
    info["bias_shape"]=save(archive,root,state,key+".bias",out/f"{prefix}_bias.bin")
    if bn:
        for source,target in [("weight","weight"),("bias","bias"),("running_mean","mean"),("running_var","var")]:
            save(archive,root,state,bn+"."+source,out/f"{prefix}_bn_{target}.bin")
    return info
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--output-dir",required=True,type=Path);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={"source_checkpoint":str(a.checkpoint.resolve()),"layers":[]}
    with zipfile.ZipFile(a.checkpoint) as z:
        data=next(x for x in z.namelist() if x.endswith("data.pkl"));root=data.split('/')[0];state=Unpickler(io.BytesIO(z.read(data))).load()["state_dict"]
        manifest["layers"].append(export_conv(z,root,state,a.output_dir,"shared","bbox_head.shared_conv.0","bbox_head.shared_conv.1"))
        for name in ["reg","height","dim","rot","hm"]:
            base=f"bbox_head.tasks.0.{name}"
            manifest["layers"].append(export_conv(z,root,state,a.output_dir,name+"_hidden",base+".0",base+".1"))
            manifest["layers"].append(export_conv(z,root,state,a.output_dir,name+"_output",base+".3",None))
    (a.output_dir/"head_weights_metadata.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"exported layers: {len(manifest['layers'])}");print(f"output: {a.output_dir}")
if __name__=="__main__": main()
