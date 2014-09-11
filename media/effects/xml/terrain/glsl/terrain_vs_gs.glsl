out ivec2 gridPosition;

void main(void)
{
  ivec2 heightMapSize = textureSize( heightMap, 0 );
  
  gridPosition.y = gl_VertexID / (heightMapSize.x - 1);
  gridPosition.x = gl_VertexID - (gridPosition.y * (heightMapSize.y - 1));
}
