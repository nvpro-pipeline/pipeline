
void main(void)
{
  vec3 t = normalize(varTangent);
  Color = vec4(t * 0.5 + 0.5, 1.0);
}
