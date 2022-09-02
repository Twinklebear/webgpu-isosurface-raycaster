# WebGPU Implicit Isosurface Raycaster

An implicit isosurface raycaster using WebGPU. Ray-isosurface intersections are computed
using the technique described by Marmitt et al.
["Fast and Accurate Ray-Voxel Intersection Techniques for Iso-Surface Ray Tracing"](http://hodad.bioen.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf),
2004.

## Live Demo

Try it out [online!](https://www.willusher.io/webgpu-isosurface-raycaster) (requires WebGPU enabled in your browser).

<img width="831" alt="bonsai" src="https://user-images.githubusercontent.com/1522476/188052052-a96a92cb-1529-44c2-b31b-d3b139c5ecbf.png">
<img width="830" alt="foot" src="https://user-images.githubusercontent.com/1522476/188052057-52a2fc00-380f-411e-9fac-30d984bde80e.png">


## Building

The project uses webpack and node to build, after cloning the repo run:

```
npm install
npm run serve
```

Then point your browser to `localhost:8080`!
