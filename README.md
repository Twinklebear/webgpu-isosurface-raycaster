# WebGPU Implicit Isosurface Raycaster

An implicit isosurface raycaster using WebGPU. Ray-isosurface intersections are computed
using the technique described by Marmitt et al.
["Fast and Accurate Ray-Voxel Intersection Techniques for Iso-Surface Ray Tracing"](http://hodad.bioen.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf),
2004.

## Live Demo

Try it out [online!]() (requires WebGPU enabled in your browser).

## Building

The project uses webpack and node to build, after cloning the repo run:

```
npm install
npm run serve
```

Then point your browser to `localhost:8080`!
