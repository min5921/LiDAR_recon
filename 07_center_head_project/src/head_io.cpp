#include "centerpoint/head.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>

namespace centerpoint { namespace {
std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("cannot open: "+path.string());
    const auto bytes=f.tellg();
    if(bytes<0 || static_cast<std::uintmax_t>(bytes)%4) throw std::runtime_error("invalid float file: "+path.string());
    std::vector<float> v(static_cast<std::size_t>(bytes)/4); f.seekg(0); f.read(reinterpret_cast<char*>(v.data()),bytes);
    if(!f) throw std::runtime_error("cannot read: "+path.string()); return v;
}
std::string read_text(const std::filesystem::path& p){ std::ifstream f(p); if(!f) throw std::runtime_error("cannot open metadata"); return {std::istreambuf_iterator<char>(f),{}}; }
int field(const std::string& s,const std::string& n){ std::smatch m; if(!std::regex_search(s,m,std::regex("\\\""+n+"\\\"\\s*:\\s*(\\d+)"))) throw std::runtime_error("missing field: "+n); return std::stoi(m[1]); }
void size_is(const std::vector<float>& v,std::size_t n,const std::string& name){ if(v.size()!=n) throw std::runtime_error(name+" size mismatch"); }
BatchNorm read_bn(const std::filesystem::path& d,const std::string& p,int c){ BatchNorm b{read_floats(d/(p+"_bn_weight.bin")),read_floats(d/(p+"_bn_bias.bin")),read_floats(d/(p+"_bn_mean.bin")),read_floats(d/(p+"_bn_var.bin"))}; size_is(b.weight,c,p); size_is(b.bias,c,p); size_is(b.mean,c,p); size_is(b.variance,c,p); return b; }
Conv read_conv(const std::filesystem::path& d,const std::string& p,int in,int out,bool bn){ Conv c; c.name=p;c.in_channels=in;c.out_channels=out;c.has_bn=bn;c.weight=read_floats(d/(p+"_weight.bin"));c.bias=read_floats(d/(p+"_bias.bin"));size_is(c.weight,static_cast<std::size_t>(out)*in*9,p);size_is(c.bias,out,p);if(bn)c.bn=read_bn(d,p,out);return c; }
}
Tensor read_rpn_output(const std::filesystem::path& d){ const auto s=read_text(d/"rpn_features_metadata.json"); Tensor t; std::smatch m; if(!std::regex_search(s,m,std::regex("\\\"shape\\\"\\s*:\\s*\\[\\s*1\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)"))) throw std::runtime_error("invalid RPN shape");t.channels=std::stoi(m[1]);t.height=std::stoi(m[2]);t.width=std::stoi(m[3]);t.values=read_floats(d/"rpn_features.bin");size_is(t.values,static_cast<std::size_t>(t.channels)*t.height*t.width,"RPN tensor");return t; }
HeadWeights read_head_weights(const std::filesystem::path& d){ HeadWeights w;w.shared=read_conv(d,"shared",384,64,true);const std::array<std::string,5> names={"reg","height","dim","rot","hm"};const std::array<int,5> outs={2,1,3,2,3};for(int i=0;i<5;++i){w.branches[i].name=names[i];w.branches[i].hidden=read_conv(d,names[i]+"_hidden",64,64,true);w.branches[i].output=read_conv(d,names[i]+"_output",64,outs[i],false);}return w;}
void write_head_output(const std::filesystem::path& d,const HeadResult& r){std::filesystem::create_directories(d);const std::array<std::string,5> names={"reg","height","dim","rot","hm"};std::ofstream meta(d/"center_head_metadata.json");meta<<"{\n  \"layout\": \"NCHW\",\n  \"elapsed_ms\": "<<r.elapsed_ms<<",\n  \"outputs\": [\n";for(int i=0;i<5;++i){const auto& t=r.outputs[i];std::ofstream f(d/(names[i]+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(t.values.data()),static_cast<std::streamsize>(t.values.size()*4));auto mm=std::minmax_element(t.values.begin(),t.values.end());std::size_t bad=0;for(float v:t.values)if(!std::isfinite(v))++bad;meta<<"    {\"name\": \""<<names[i]<<"\", \"shape\": [1, "<<t.channels<<", "<<t.height<<", "<<t.width<<"], \"min\": "<<*mm.first<<", \"max\": "<<*mm.second<<", \"non_finite\": "<<bad<<"}"<<(i==4?"\n":",\n");}meta<<"  ]\n}\n";}
}
