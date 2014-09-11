out ivec2 vs_gridPosition;

out float tc_tessLevelMax;

void main(void)
{
  tc_tessLevelMax = 4.0f;
  
  ivec2 heightMapSizeOrig = textureSize( heightMap, 0 ) - ivec2(1);
  ivec2 heightMapSize = heightMapSizeOrig / ivec2(tc_tessLevelMax);
  
  
  vs_gridPosition.y = gl_VertexID / (heightMapSize.x);
  vs_gridPosition.x = gl_VertexID - (vs_gridPosition.y * (heightMapSize.y));
}
