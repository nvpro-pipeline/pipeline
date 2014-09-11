// number of CPs in patch
layout (vertices = 1) out;

// from VS (use empty modifier [] so we can say anything)
in ivec2 vs_gridPosition[];

// to evluation shader. will be used to guide positioning of generated points
out ivec2 tc_gridPosition[];
 
in float tc_tessLevelMax[];
out float te_tessLevelMax[];

void main () {
  te_tessLevelMax[gl_InvocationID] = tc_tessLevelMax[gl_InvocationID];

  tc_gridPosition[gl_InvocationID] = vs_gridPosition[gl_InvocationID] * ivec2(te_tessLevelMax[0]);
  
  // Calculate the tessellation levels
  gl_TessLevelInner[0] = te_tessLevelMax[0]; // number of nested primitives to generate
  gl_TessLevelInner[1] = te_tessLevelMax[0]; // number of nested primitives to generate
  gl_TessLevelOuter[0] = te_tessLevelMax[0]; // times to subdivide first side
  gl_TessLevelOuter[1] = te_tessLevelMax[0]; // times to subdivide second side
  gl_TessLevelOuter[2] = te_tessLevelMax[0]; // times to subdivide third side
  gl_TessLevelOuter[3] = te_tessLevelMax[0]; // times to subdivide third side
}
