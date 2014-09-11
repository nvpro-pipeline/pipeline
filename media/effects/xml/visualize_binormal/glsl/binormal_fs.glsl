
void main(void)
{
  vec3 b = normalize(varBinormal);
  Color = vec4(b * 0.5 + 0.5, 1.0);
}
