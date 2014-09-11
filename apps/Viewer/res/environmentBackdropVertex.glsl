out gl_PerVertex
{
  vec4 gl_Position;
};

void main(void)
{
  gl_Position  = attrPosition;
  varTexCoord0 = attrTexCoord2.xyz;
}
