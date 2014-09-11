out gl_PerVertex
{
  vec4 gl_Position;
};

void main(void)
{
  varTexCoord0 = attrTexCoord0;
  gl_Position  = attrPosition;
}
