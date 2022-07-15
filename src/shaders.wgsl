// Reduce clutter/keyboard pain
type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;
type int2 = vec2<i32>;
type int3 = vec3<i32>;
type int4 = vec4<i32>;
type bool3 = vec3<bool>;

struct VertexInput {
    @location(0) position: float3,
};

struct VertexOutput {
    @builtin(position) position: float4,
    @location(0) transformed_eye: float3,
    @location(1) ray_dir: float3,
};

struct ViewParams {
    proj_view: mat4x4<f32>,
    // Not sure on WGSL padding/alignment rules for blocks,
    // just assume align/pad to vec4
    eye_pos: float4,
    //volume_scale: float4;
    volume_dims: int4,
    isovalue: f32,
};

struct GridIterator {
    grid_dims: int3,
    grid_step: int3,
    t_delta: float3,

    cell: int3,
    t_max: float3,
    t: f32,
};

@group(0) @binding(0)
var<uniform> view_params: ViewParams;

@group(0) @binding(1)
var volume: texture_3d<f32>;

@group(0) @binding(2)
var colormap: texture_2d<f32>;

@group(0) @binding(3)
var tex_sampler: sampler;

fn outside_grid(p: int3, grid_dims: int3) -> bool {
    return any(p < int3(0)) || any(p >= grid_dims);
}

// Initialize the grid traversal state. All positions/directions passed must be in the
// grid coordinate system where a grid cell is 1^3 in size.
fn init_grid_iterator(ray_org: float3, ray_dir: float3, t: f32, grid_dims: int3) -> GridIterator {
    var grid_iter: GridIterator;
    grid_iter.grid_dims = grid_dims;
    grid_iter.grid_step = int3(sign(ray_dir));

    let inv_ray_dir = 1.0 / ray_dir;
    grid_iter.t_delta = abs(inv_ray_dir);

	var p = (ray_org + t * ray_dir);
    p = clamp(p, float3(0), float3(grid_dims - 1));
    let cell = floor(p);
    let t_max_neg = (cell - ray_org) * inv_ray_dir;
    let t_max_pos = (cell + float3(1) - ray_org) * inv_ray_dir;

    // Pick between positive/negative t_max based on the ray sign
    let is_neg_dir = ray_dir < float3(0);
    grid_iter.t_max = select(t_max_pos, t_max_neg, is_neg_dir);

    grid_iter.cell = int3(cell);

    grid_iter.t = t;

    return grid_iter;
}

fn grid_iterator_next_cell(iter: ptr<function, GridIterator, read_write>,
                           cell_t_range: ptr<function, float2, read_write>,
                           cell_id: ptr<function, int3, read_write>) -> bool {
    // Please add arrow operator or something equivalent for it, this is terrible to type
    // and to read
    if (outside_grid((*iter).cell, (*iter).grid_dims)) {
        return false;
    }
    // Return the current cell range and ID to the caller
    (*cell_t_range).x = (*iter).t;
    (*cell_t_range).y = min((*iter).t_max.x, min((*iter).t_max.y, (*iter).t_max.z));
    *cell_id = (*iter).cell;
    if ((*cell_t_range).y < (*cell_t_range).x) {
        return false;
    }

    // Move the iterator to the next cell we'll traverse
    (*iter).t = (*cell_t_range).y;
    if ((*iter).t == (*iter).t_max.x) {
        (*iter).cell.x += (*iter).grid_step.x;
        (*iter).t_max.x += (*iter).t_delta.x;
    } else if ((*iter).t == (*iter).t_max.y) {
        (*iter).cell.y += (*iter).grid_step.y;
        (*iter).t_max.y += (*iter).t_delta.y;
    } else {
        (*iter).cell.z += (*iter).grid_step.z;
        (*iter).t_max.z += (*iter).t_delta.z;
    }
    return true;
}

// Load the vertex values for the dual cell 'cell_id's vertices
// Vertex values will be returned in the order:
// [v000, v100, v110, v010, v001, v101, v111, v011]
// v000 = cell_id
fn load_dual_cell(cell_id: int3, values: ptr<function, array<f32, 8>, read_write>) -> float2 {
    let index_to_vertex = array<int3, 8>(
        int3(0, 0, 0), // v000 = 0
        int3(1, 0, 0), // v100 = 1
        int3(0, 1, 0), // v010 = 2
        int3(1, 1, 0), // v110 = 3
        int3(0, 0, 1), // v001 = 4
        int3(1, 0, 1), // v101 = 5
        int3(0, 1, 1), // v011 = 6
        int3(1, 1, 1)  // v111 = 7
    );

    var cell_range = float2(1e20f, -1e20f);
    for (var i = 0; i < 8; i++) { 
        let v = cell_id + index_to_vertex[i];
        //var val = textureSampleLevel(volume, tex_sampler, p, 0.0).r;
        var val = textureLoad(volume, v, 0).r;
        (*values)[i] = val;
        cell_range.x = min(cell_range.x, val);
        cell_range.y = max(cell_range.y, val);
    }
    return cell_range;
}

// Compute the polynomial for the cell with the given vertex values
fn compute_polynomial(p: float3,
                      dir: float3,
                      v000: float3,
                      values: ptr<function, array<f32, 8>, read_write>) -> float4
{
    let v111 = v000 + float3(1);
    // Note: Grid voxels sizes are 1^3
    let a = array<float3, 2>(v111 - p, p - v000);
    let b = array<float3, 2>(dir, -dir);

    var poly = float4(0);
    poly.w -= view_params.isovalue;
    for (var k = 0; k < 2; k++) {
        for (var j = 0; j < 2; j++) {
            for (var i = 0; i < 2; i++) {
                let val = (*values)[i + 2 * (j + 2 * k)];

                poly.x += b[i].x * b[j].y * b[k].z * val;

                poly.y += (a[i].x * b[j].y * b[k].z +
                        b[i].x * a[j].y * b[k].z +
                        b[i].x * b[j].y * a[k].z) * val;

                poly.z += (b[i].x * a[j].y * a[k].z +
                        a[i].x * b[j].y * a[k].z +
                        a[i].x * a[j].y * b[k].z) * val;

                poly.w += a[i].x * a[j].y * a[k].z * val;
            }
        }
    }
    return poly;
}

fn evaluate_polynomial(poly: float4, t: f32) -> f32 {
    return poly.x * pow(t, 3.f) + poly.y * pow(t, 2.f) + poly.z * t + poly.w;
}

// Returns true if the quadratic has real roots
fn solve_quadratic(poly: float3, roots: ptr<function, array<f32, 2>, read_write>) -> bool {
    // Check for case when poly is just Bt + c = 0
    if (poly.x == 0) {
        (*roots)[0] = -poly.z/poly.y;
        (*roots)[1] = -poly.z/poly.y;
        return true;
    }
    var discriminant = pow(poly.y, 2.f) - 4.f * poly.x * poly.z;
    if (discriminant < 0.f) {
        return false;
    }
    discriminant = sqrt(discriminant);
    var r = 0.5f * float2(-poly.y + discriminant, -poly.y - discriminant) / poly.x;
    (*roots)[0] = min(r.x, r.y);
    (*roots)[1] = max(r.x, r.y);
    return true;
}

// Trilinear interpolation at the given point within the cell with its origin at v000
// (origin = bottom-left-near point)
fn trilinear_interpolate_in_cell(p: float3,
                                 v000: float3,
                                 values: ptr<function, array<f32, 8>, read_write>) -> f32 {
    let diff = clamp(p, v000, v000 + float3(1.0)) - v000;
    // Interpolate across x, then y, then z, and return the value normalized between 0 and 1
    let c00 = (*values)[0] * (1.f - diff.x) + (*values)[1] * diff.x;
    let c01 = (*values)[4] * (1.f - diff.x) + (*values)[5] * diff.x;
    let c10 = (*values)[2] * (1.f - diff.x) + (*values)[3] * diff.x;
    let c11 = (*values)[6] * (1.f - diff.x) + (*values)[7] * diff.x;
    let c0 = c00 * (1.f - diff.y) + c10 * diff.y;
    let c1 = c01 * (1.f - diff.y) + c11 * diff.y;
    return c0 * (1.f - diff.z) + c1 * diff.z;
}

fn marmitt_intersect(ray_org: float3,
                     ray_dir: float3,
                     v000: float3,
                     values: ptr<function, array<f32, 8>, read_write>,
                     t_prev: f32,
                     t_next: f32,
                     t_hit: ptr<function, f32, read_write>) -> bool
{
    // The text seems to not say explicitly, but I think it is required to have
    // the ray "origin" within the cell for the cell-local coordinates for a to
    // be computed properly. So here I set the cell_p to be at the midpoint of the
    // ray's overlap with the cell, which makes it easy to compute t_in/t_out and
    // avoid numerical issues with cell_p being right at the edge of the cell.
    let cell_p = ray_org + ray_dir * (t_prev + (t_next - t_prev) * 0.5f);
    var t_in = -(t_next - t_prev) * 0.5f;
    var t_out = (t_next - t_prev) * 0.5f;
    let poly = compute_polynomial(cell_p, ray_dir, v000, values);

    var f_in = evaluate_polynomial(poly, t_in);
    var f_out = evaluate_polynomial(poly, t_out);
    var roots = array<f32, 2>(0, 0);
    // TODO: Seeming to get some holes in the surface with the Marmitt intersector
    if (solve_quadratic(float3(3.f * poly.x, 2.f * poly.y, poly.z), &roots)) {
        if (roots[0] >= t_in && roots[0] <= t_out) {
            let f_root0 = evaluate_polynomial(poly, roots[0]);
            if (sign(f_root0) == sign(f_in)) {
                t_in = roots[0];
                f_in = f_root0;
            } else {
                t_out = roots[0];
                f_out = f_root0;
            }
        }
        if (roots[1] >= t_in && roots[1] <= t_out) {
            let f_root1 = evaluate_polynomial(poly, roots[1]);
            if (sign(f_root1) == sign(f_in)) {
                t_in = roots[1];
                f_in = f_root1;
            } else {
                t_out = roots[1];
                f_out = f_root1;
            }
        }
    }
    // If the signs aren't equal we know there's an intersection in the cell
    if (sign(f_in) != sign(f_out)) {
        // Find the intersection via repeated linear interpolation
        for (var i = 0; i < 10; i++) {
            let t = t_in + (t_out - t_in) * (-f_in) / (f_out - f_in);
            let f_t = evaluate_polynomial(poly, t);
            if (sign(f_t) == sign(f_in)) {
                t_in = t;
                f_in = f_t;
            } else {
                t_out = t;
                f_out = f_t;
            }
        }
        *t_hit = t_in + (t_out - t_in) * (-f_in) / (f_out - f_in);
        // Return t_hit relative to vol_eye
        let hit_p = cell_p + ray_dir * *t_hit;
        *t_hit = length(hit_p - ray_org) / length(ray_dir);
        return true;
    }
    return false;
}

fn compute_gradient(p: float3,
                    v000: float3,
                    values: ptr<function, array<f32, 8>, read_write>) -> float3 {
    let v111 = v000 + float3(1);
    let deltas = array<float3, 3>(
        float3(0.1, 0.0, 0.0),
        float3(0.0, 0.1, 0.0),
        float3(0.0, 0.0, 0.1)
    );
    var n = float3(0);
    // TODO: Pretty sure it's the clamping that is producing at least the artifacts
    // at the cell boundaries. Not sure about the bad looking gradient or surface in
    // other areas though
    n.x = trilinear_interpolate_in_cell(min(p + deltas[0], v111), v000, values)
          - trilinear_interpolate_in_cell(max(p - deltas[0], v000), v000, values);
    n.y = trilinear_interpolate_in_cell(min(p + deltas[1], v111), v000, values)
          - trilinear_interpolate_in_cell(max(p - deltas[1], v000), v000, values);
    n.z = trilinear_interpolate_in_cell(min(p + deltas[2], v111), v000, values)
          - trilinear_interpolate_in_cell(max(p - deltas[2], v000), v000, values);
    return normalize(n);
}

fn intersect_box(orig: float3, dir: float3) -> float2 {
	var box_min = float3(0.0);
	var box_max = float3(1.0);
	var inv_dir = 1.0 / dir;
	var tmin_tmp = (box_min - orig) * inv_dir;
	var tmax_tmp = (box_max - orig) * inv_dir;
	var tmin = min(tmin_tmp, tmax_tmp);
	var tmax = max(tmin_tmp, tmax_tmp);
	var t0 = max(tmin.x, max(tmin.y, tmin.z));
	var t1 = min(tmax.x, min(tmax.y, tmax.z));
	return float2(t0, t1);
}

fn linear_to_srgb(x: f32) -> f32 {
	if (x <= 0.0031308) {
		return 12.92 * x;
	}
	return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = vert.position;
    out.position = view_params.proj_view * float4(pos, 1.0);
    out.transformed_eye = view_params.eye_pos.xyz;
    out.ray_dir = pos - out.transformed_eye;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) float4 {
    var ray_dir = normalize(in.ray_dir);

    var t_hit = intersect_box(in.transformed_eye, ray_dir);
    if (t_hit.x > t_hit.y) {
        discard;
    }
    t_hit.x = max(t_hit.x, 0.0);

    // Scale the eye and ray direction from the 1^3 box to the volume grid
    ray_dir *= float3(view_params.volume_dims.xyz);
    var ray_org = in.transformed_eye * float3(view_params.volume_dims.xyz) - float3(0.5);
    let dual_grid_dims = view_params.volume_dims.xyz - int3(1);

    // TODO: For isosurface we need to translate onto the dual grid and use the dual grid dimensions
    var iter = init_grid_iterator(ray_org, ray_dir, t_hit.x, dual_grid_dims);

    var color = float4(0);
    var cell_id = int3(0);
    var cell_t_range = float2(0);
    while (grid_iterator_next_cell(&iter, &cell_t_range, &cell_id)) {
        var vertex_values: array<f32, 8>;
        let cell_range = load_dual_cell(cell_id, &vertex_values);
        // TODO: Seems like the range doesn't quite match up with what it should be,
        // having this filter here results in some errors on Neghip, while the actual
        // surface does look right
        if (true) {//view_params.isovalue >= cell_range.x && view_params.isovalue <= cell_range.y) {
            var t_hit: f32;
            let hit = marmitt_intersect(ray_org,
                                        ray_dir,
                                        float3(cell_id),
                                        &vertex_values,
                                        cell_t_range.x,
                                        cell_t_range.y,
                                        &t_hit);
            if (hit) {
                color = float4(float3(cell_id) / float3(dual_grid_dims), 1.0);
                /*
                let hit_p = ray_org + t_hit * ray_dir;
                var normal = compute_gradient(hit_p, float3(cell_id), &vertex_values);
                if (dot(normal, ray_dir) > 0.0) {
                    normal = -normal;
                }
                color = float4((normal + float3(1.0)) * 0.5, 1.0);
                */
                break;
            }
        }
        /*
        // Old volume renderer
        var t = 0.5 * (cell_t_range.x + cell_t_range.y);
        var p = (ray_org + t * ray_dir) / float3(view_params.volume_dims.xyz);
        var val = textureSampleLevel(volume, tex_sampler, p, 0.0).r;
        var val_color = float4(textureSampleLevel(colormap, tex_sampler, float2(val, 0.5), 0.0).rgb, val);
        // Opacity correction
        val_color.a = 1.0 - pow(1.0 - val_color.a, (cell_t_range.y - cell_t_range.x) / 1.7);
        val_color.a = clamp(val_color.a * 50.0, 0.0, 1.0);
        // WGSL can't do left hand size swizzling!?!?
        // https://github.com/gpuweb/gpuweb/issues/737 
        // That's ridiculous for a shader language.
        var tmp = color.rgb + (1.0 - color.a) * val_color.a * val_color.xyz; 
        color.r = tmp.r;
        color.g = tmp.g;
        color.b = tmp.b;
        color.a = color.a + (1.0 - color.a) * val_color.a;
        if (color.a >= 0.95) {
            break;
        }
        */
    }

    color.r = linear_to_srgb(color.r);
    color.g = linear_to_srgb(color.g);
    color.b = linear_to_srgb(color.b);
    return color;
}

