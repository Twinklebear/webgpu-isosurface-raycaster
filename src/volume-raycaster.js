import {ArcballCamera} from "arcball_camera";
import {Controller} from "ez_canvas_controller";
import {mat4, vec3} from "gl-matrix";

import shaderCode from "./shaders.wgsl";
import {
    fetchVolume,
    fillSelector,
    getCubeMesh,
    getVolumeDimensions,
    uploadVolume,
    volumes
} from "./volume.js";

(async () => {
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    var adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }
    var device = await adapter.requestDevice();

    // Get a context to display our rendered image on the canvas
    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("webgpu");

    // Setup shader modules
    var shaderModule = device.createShaderModule({code: shaderCode});
    var compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        var hadError = false;
        console.log("Shader compilation log:");
        for (var i = 0; i < compilationInfo.messages.length; ++i) {
            var msg = compilationInfo.messages[i];
            console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
            hadError = hadError || msg.type == "error";
        }
        if (hadError) {
            console.log("Shader failed to compile");
            return;
        }
    }

    const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 2.5);
    const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
    const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

    const cube = getCubeMesh();

    // Upload cube to use to trigger raycasting of the volume
    var vertexBuffer = device.createBuffer({
        size: cube.vertices.length * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(cube.vertices);
    vertexBuffer.unmap();

    var indexBuffer = device.createBuffer(
        {size: cube.indices.length * 4, usage: GPUBufferUsage.INDEX, mappedAtCreation: true});
    new Uint16Array(indexBuffer.getMappedRange()).set(cube.indices);
    indexBuffer.unmap();

    // Create a buffer to store the view parameters
    const viewParamsSize = 4 * (16 + 4 + 4 + 1) + 12;
    var viewParamsBuffer = device.createBuffer(
        {size: viewParamsSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});

    var volumePicker = document.getElementById("volumeList");
    var isovalueSlider = document.getElementById("isovalue");
    isovalueSlider.value = 128;

    fillSelector(volumePicker, volumes);

    // Fetch and upload the volume
    var volumeName = "Bonsai";
    if (window.location.hash) {
        var linkedDataset = decodeURI(window.location.hash.substring(1));
        if (linkedDataset in volumes) {
            volumePicker.value = linkedDataset;
            volumeName = linkedDataset;
        } else {
            alert(`Linked to invalid data set ${linkedDataset}`);
            return;
        }
    }

    var volumeDims = getVolumeDimensions(volumes[volumeName]);
    var volumeTexture =
        await fetchVolume(volumes[volumeName])
            .then((volumeData) => {return uploadVolume(device, volumeDims, volumeData);});

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    context.configure({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
        alphaMode: "premultiplied"
    });

    var bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {type: "uniform"}
            },
            {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {viewDimension: "3d"}}
        ]
    });

    // Create render pipeline
    var layout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

    var vertexState = {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: [{
            arrayStride: 3 * 4,
            attributes: [{format: "float32x3", offset: 0, shaderLocation: 0}]
        }]
    };

    var fragmentState = {
        module: shaderModule,
        entryPoint: "fragment_main",
        targets: [{
            format: swapChainFormat,
            blend: {
                color: {srcFactor: "one", dstFactor: "one-minus-src-alpha"},
                alpha: {srcFactor: "one", dstFactor: "one-minus-src-alpha"}
            }
        }]
    };

    var renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertex: vertexState,
        fragment: fragmentState,
        primitive: {
            topology: "triangle-strip",
            stripIndexFormat: "uint16",
            cullMode: "front",
        }
    });

    var renderPassDesc = {
        colorAttachments: [{
            view: undefined,
            loadOp: "clear",
            storeOp: "store",
            clearValue: [1.0, 1.0, 1.0, 1]
        }]
    };

    var camera = new ArcballCamera(defaultEye, center, up, 2, [canvas.width, canvas.height]);
    var proj = mat4.perspective(
        mat4.create(), 50 * Math.PI / 180.0, canvas.width / canvas.height, 0.1, 100);
    var projView = mat4.create();

    // Register mouse and touch listeners
    var controller = new Controller();
    controller.mousemove = function (prev, cur, evt) {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function (amt) {
        camera.zoom(amt * 0.1);
    };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function (drag) {
        camera.pan(drag);
    };
    controller.registerForCanvas(canvas);

    var bindGroupEntries = [
        {binding: 0, resource: {buffer: viewParamsBuffer}},
        {binding: 1, resource: volumeTexture.createView()}
    ];
    var bindGroup =
        device.createBindGroup({layout: bindGroupLayout, entries: bindGroupEntries});

    var upload = device.createBuffer({
        size: viewParamsSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false
    });

    const render = async () => {
        // Fetch a new volume if a new one was selected
        if (volumeName != volumePicker.value) {
            volumeName = volumePicker.value;
            history.replaceState(history.state, "", "#" + volumeName);

            volumeDims = getVolumeDimensions(volumes[volumeName]);

            volumeTexture = await fetchVolume(volumes[volumeName]).then((volumeData) => {
                return uploadVolume(device, volumeDims, volumeData);
            });

            bindGroupEntries[1].resource = volumeTexture.createView();
            bindGroup =
                device.createBindGroup({layout: bindGroupLayout, entries: bindGroupEntries});
        }

        // Update camera buffer
        projView = mat4.mul(projView, proj, camera.camera);

        {
            await upload.mapAsync(GPUMapMode.WRITE)
            var eyePos = camera.eyePos();
            var map = upload.getMappedRange();
            var fmap = new Float32Array(map);
            var imap = new Int32Array(map);

            fmap.set(projView);
            fmap.set(eyePos, projView.length);
            imap.set(volumeDims, projView.length + 4);
            fmap.set([isovalueSlider.value / 255.0], projView.length + 8);
            upload.unmap();
        }

        var commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(upload, 0, viewParamsBuffer, 0, viewParamsSize);

        renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();
        var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, "uint16");
        renderPass.draw(cube.vertices.length / 3, 1, 0, 0);

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
})();
